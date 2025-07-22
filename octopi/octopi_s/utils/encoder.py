import torch
from torch import nn
from transformers import CLIPVisionModel
from transformers.models.clip.modeling_clip import CLIPModel, CLIPVisionTransformer, CLIPTextTransformer, CLIPEncoder, CLIPEncoderLayer, _create_4d_causal_attention_mask, _prepare_4d_attention_mask
from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
from typing import Any, Optional, Tuple, Union
from .dataset import get_frames, get_dataset_sensor_type, get_image_transforms
import os, json, yaml, warnings


class PromptLearningCLIPEncoderLayer(CLIPEncoderLayer):
    def __init__(self, config, configs, text_layer, layer_idx, prompt_depth):
        super().__init__(config)
        self.text_layer = text_layer
        self.layer_idx = layer_idx
        self.prompt_depth = prompt_depth
        if layer_idx != 0:
            self.add_prompt = layer_idx < prompt_depth
            if self.text_layer:
                self.n_ctx_text = configs["num_context_text"]
                if self.add_prompt:
                    ctx_vectors = torch.randn((self.n_ctx_text, configs["dim_context_text"])) * 0.02
            else: # NOTE: No sensor context
                self.n_ctx_visual = configs["num_context_vision"]
                if self.add_prompt:
                    ctx_vectors = torch.randn((self.n_ctx_visual, configs["dim_context_vision"])) * 0.02
            if self.add_prompt:
                self.VPT_shallow = nn.Parameter(ctx_vectors)
        else:
            self.add_prompt = False
            if self.text_layer:
                self.n_ctx_text = configs["num_context_text"]
            else:
                self.n_ctx_visual = configs["num_context_vision"]
        if layer_idx != config.num_hidden_layers - 1:
            self.add_gate_value = True
            self.VPT_gamma = nn.Parameter(torch.tensor(configs["gate_prior"], requires_grad=True))
        else:
            self.add_gate_value = False
            

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        causal_attention_mask: torch.Tensor,
        output_attentions: Optional[bool] = False
    ) -> Tuple[torch.FloatTensor]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
                `(config.encoder_attention_heads,)`.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        # Add learnable prompts
        if not self.text_layer:
            self.n_ctx = self.n_ctx_visual
        if self.add_gate_value:
            if not self.text_layer:
                prompt_before_block = hidden_states[:, hidden_states.shape[1] - self.n_ctx:, :]
            else:
                prompt_before_block = hidden_states[:, 1:1+self.n_ctx_text:, :]
        if self.add_prompt:
            if not self.text_layer:
                # hidden_states -> (N, L, DIM=1024)
                # Remove the outputs produced by learnable tokens of previous layer
                prefix = hidden_states[:, :hidden_states.shape[1] - self.n_ctx, :] # (N, L, DIM)
                visual_context = self.VPT_shallow.expand(hidden_states.shape[0], -1, -1) # (N, n_ctx_visual, DIM)
                hidden_states = torch.cat([prefix, visual_context], dim=1) # (N, L + n_ctx, DIM)
            else:
                # hidden_states -> (N, L, DIM=768) --> NOTE: 6 belongs to "A tactile sensor video of"
                # Remove the outputs produced by learnable tokens of previous layer
                prefix = hidden_states[:, :1, :]
                suffix = hidden_states[:, 1 + self.n_ctx_text:, :]
                textual_context = self.VPT_shallow.expand(hidden_states.shape[0], -1, -1)
                # textual_context = self.VPT_shallow.expand(hidden_states.shape[1], -1, -1).permute(1, 0, 2) # .half()
                hidden_states = torch.cat([prefix, textual_context, suffix], dim=1)
        else:
            # First layer to remove learnable prompts
            if self.layer_idx == self.prompt_depth:
                if not self.text_layer:
                    prefix = hidden_states[:, :hidden_states.shape[1] - self.n_ctx, :]
                    hidden_states = prefix
                else:
                    prefix = hidden_states[:, :1, :]
                    suffix = hidden_states[:, 1 + self.n_ctx_text:, :]
                    hidden_states = torch.cat([prefix, suffix], dim=1)

        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
        )
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        if self.add_prompt:
            if self.add_gate_value:
                gate = self.VPT_gamma.sigmoid()
                if not self.text_layer:
                    prompt_after_block = hidden_states[:, hidden_states.shape[1] - self.n_ctx:, :]
                else:
                    prompt_after_block = hidden_states[:, 1:1+self.n_ctx_text:, :]
                gated_prompt = gate * prompt_after_block + (1 - gate) * prompt_before_block
                if not self.text_layer:
                    hidden_states = torch.cat([
                        hidden_states[:, :hidden_states.shape[1] - self.n_ctx, :],
                        gated_prompt
                    ], dim=1)
                else:
                    prefix = hidden_states[:, :1, :]
                    suffix = hidden_states[:, 1 + self.n_ctx_text:, :]
                    hidden_states = torch.cat([prefix, gated_prompt, suffix], dim=1)

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (attn_weights,)
        return outputs
        

class PromptLearningCLIPEncoder(CLIPEncoder):
    def __init__(self, config, configs, text_layer, prompt_depth):
        super().__init__(config)
        self.config = config
        if prompt_depth == -1:
            prompt_depth = config.num_hidden_layers
        self.layers = nn.ModuleList([PromptLearningCLIPEncoderLayer(config, configs, text_layer, layer_idx, prompt_depth) for layer_idx in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def forward(
        self,
        inputs_embeds,
        attention_mask: Optional[torch.Tensor] = None,
        causal_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        r"""
        Args:
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert `input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            causal_attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Causal mask for the text model. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        hidden_states = inputs_embeds
        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    encoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    causal_attention_mask,
                    output_attentions,
                )
            else:
                layer_outputs = encoder_layer(
                    hidden_states,
                    attention_mask,
                    causal_attention_mask,
                    output_attentions=output_attentions,
                )

            hidden_states = layer_outputs[0]
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)
        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)
        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions
        )


class PromptLearningCLIPVisionTransformer(CLIPVisionTransformer):
    def __init__(self, config, configs, text_layer):
        super().__init__(config)
        self.encoder = PromptLearningCLIPEncoder(config, configs, text_layer, configs["prompt_depth_vision"])
        self.configs = configs
        if configs["prompt_depth_vision"] == 0:
            self.VPT_shallow = False
        else:
            self.VPT_shallow = True
        if self.VPT_shallow:
            # Learnable prompt tokens as input in the first layer
            n_ctx = configs["num_context_vision"]
            ctx_vectors = torch.randn((n_ctx, configs["dim_context_vision"])) * 0.02
            self.VPT = nn.Parameter(ctx_vectors)
        self.prompt_till_layer_visual = configs["prompt_depth_vision"]
    
    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        Returns:

        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")
        hidden_states = self.embeddings(pixel_values) # (N, L, D)
        if self.VPT_shallow:
            # Learnable prompt tokens as input in the first layer
            visual_ctx = self.VPT.expand(hidden_states.shape[0], -1, -1) # .half() # (N, n_ctx_vision, D)
            hidden_states = torch.cat([hidden_states, visual_ctx], dim=1)
        else:
            assert self.prompt_till_layer_visual == 0
        hidden_states = self.pre_layrnorm(hidden_states)
        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        last_hidden_state = encoder_outputs[0]
        pooled_output = last_hidden_state[:, 0, :]
        pooled_output = self.post_layernorm(pooled_output)
        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]
        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class PromptLearningCLIPTextTransformer(CLIPTextTransformer):
    def __init__(self, config, configs, text_layer):
        super().__init__(config)
        self.encoder = PromptLearningCLIPEncoder(config, configs, text_layer, configs["prompt_depth_text"])
        if configs["prompt_depth_text"] == 0:
            self.VPT_shallow = False
        else:
            self.VPT_shallow = True
        if self.VPT_shallow:
            # Learnable prompt tokens as input in the first layer
            n_ctx = configs["num_context_text"]
            self.n_ctx = n_ctx
            ctx_vectors = torch.randn((n_ctx, configs["dim_context_text"])) * 0.02
            self.VPT = nn.Parameter(ctx_vectors)
        self.prompt_till_layer_text = configs["prompt_depth_text"]
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        Returns:

        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if input_ids is None:
            raise ValueError("You have to specify input_ids")
        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])
        hidden_states = self.embeddings(input_ids=input_ids, position_ids=position_ids) # (N, L, D)?
        # print("Text hidden state shape:", hidden_states)
        if self.VPT_shallow:
            # Learnable prompt tokens as input in the first layer
            text_ctx = self.VPT.expand(hidden_states.shape[0], -1, -1) # .half() # (N, n_ctx, D)
            prefix = hidden_states[:, :1, :]
            suffix = hidden_states[:, 1 + self.n_ctx:, :]
            hidden_states = torch.cat([prefix, text_ctx, suffix], dim=1) # (N, L + n_ctx, D)
            # hidden_states = torch.cat([hidden_states, text_ctx], dim=1) # (N, L + n_ctx, D)
        else:
            assert self.prompt_till_layer_text == 0
        # CLIP's text model uses causal mask, prepare it here.
        # https://github.com/openai/CLIP/blob/cfcffb90e69f37bf2ff1e988237a0fbe41f33c04/clip/model.py#L324
        causal_attention_mask = _create_4d_causal_attention_mask(
            input_shape, hidden_states.dtype, device=hidden_states.device
        )
        # expand attention_mask
        if attention_mask is not None and not self._use_flash_attention_2:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            attention_mask = _prepare_4d_attention_mask(attention_mask, hidden_states.dtype)
        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        last_hidden_state = encoder_outputs[0]
        last_hidden_state = self.final_layer_norm(last_hidden_state)
        if self.eos_token_id == 2:
            # The `eos_token_id` was incorrect before PR #24773: Let's keep what have been done here.
            # A CLIP model with such `eos_token_id` in the config can't work correctly with extra new tokens added
            # ------------------------------------------------------------
            # text_embeds.shape = [batch_size, sequence_length, transformer.width]
            # take features from the eot embedding (eot_token is the highest number in each sequence)
            # casting to torch.int for onnx compatibility: argmax doesn't support int64 inputs with opset 14
            pooled_output = last_hidden_state[
                torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device),
                input_ids.to(dtype=torch.int, device=last_hidden_state.device).argmax(dim=-1),
            ]
        else:
            # The config gets updated `eos_token_id` from PR #24773 (so the use of exta new tokens is possible)
            pooled_output = last_hidden_state[
                torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device),
                # We need to get the first position of `eos_token_id` value (`pad_token_ids` might equal to `eos_token_id`)
                # Note: we assume each sequence (along batch dim.) contains an  `eos_token_id` (e.g. prepared by the tokenizer)
                (input_ids.to(dtype=torch.int, device=last_hidden_state.device) == self.eos_token_id)
                .int()
                .argmax(dim=-1),
            ]
        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]
        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class PromptLearningCLIPModel(CLIPModel):
    def __init__(self, config, configs):
        super().__init__(config)
        text_config = config.text_config
        vision_config = config.vision_config
        self.text_model = PromptLearningCLIPTextTransformer(text_config, configs, text_layer=True)
        self.vision_model = PromptLearningCLIPVisionTransformer(vision_config, configs, text_layer=False)
        # Initialize weights and apply final processing
        self.post_init()

    
class ViFiCLIP(nn.Module):
    def __init__(self, clip_model, freeze_text_encoder, use_positional_embeds):
        super().__init__()
        self.clip_model = clip_model
        if freeze_text_encoder:
            for name, param in self.clip_model.named_parameters():
                if "text_model" in name:
                    param.requires_grad_(False)
        self.use_positional_embeds = use_positional_embeds
        self.logit_scale_tactile = nn.Parameter(torch.log(torch.tensor(1/0.07)))
        self.logit_scale_text = nn.Parameter(torch.log(torch.tensor(1/0.07)))

    def forward(self, frames, texts, attention_masks):
        # video
        b, l, c, h, w = frames.shape # (b, l, c, h, w)
        frames = frames.reshape(b * l, c, h, w) # (b * l, c, h, w)
        frame_embeds = self.clip_model.vision_model(frames)
        frame_features = frame_embeds.pooler_output # (b * l, patch_embed_size)
        # pooled_output = self.clip_model.visual_projection(vision_outputs[1])
        _, patch_embed_size = frame_features.shape
        frame_features = frame_features.reshape(b, l, patch_embed_size) # (b, l, patch_embed_size)
        video_features = frame_features.mean(dim=1, keepdim=False) # (b, patch_embed_size)
        video_features = video_features / video_features.norm(p=2, dim=-1, keepdim=True)
        if texts is not None:
            # text
            text_outputs = self.clip_model.text_model(texts, attention_mask=attention_masks)
            text_features = text_outputs.pooler_output
            # pooled_output = text_outputs[1]
            # text_features = self.clip_model.text_projection(pooled_output)
            text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
        else:
            text_features = None
        logits_per_text = None
        logits_per_image = None
        return video_features, text_features, logits_per_image, logits_per_text
    

class CLIPVisionEncoder(nn.Module):
    def __init__(self, clip_model):
        super(CLIPVisionEncoder, self).__init__()
        self.model = CLIPVisionModel.from_pretrained(clip_model)

    def forward(self, frames):
        b, l, c, h, w = frames.shape # (b, l, c, h, w)
        frames = frames.reshape(b * l, c, h, w) # (b * l, c, h, w)
        frame_embeds = self.model(frames, output_hidden_states=True)
        frame_features = frame_embeds.pooler_output
        _, patch_embed_size = frame_features.shape
        frame_features = frame_features.reshape(b, l, patch_embed_size) # (b, l, patch_embed_size)
        return frame_features
    

class Adapter(nn.Module):
    def __init__(self, input_size, output_size, residual_ratio):
        super(Adapter, self).__init__()
        self.act = nn.GELU()
        self.residual_ratio = residual_ratio
        self.rfc = nn.Sequential(
            nn.Linear(input_size, 512),
            self.act,
            nn.Linear(512, input_size),
        )
        for name, param in self.rfc.named_parameters():
            if "weight" in name:
                torch.nn.init.trunc_normal_(param, std=1e-3)
            elif "bias" in name:
                torch.nn.init.zeros_(param)
        if input_size != output_size:
            self.align = nn.Sequential(
                self.act,
                nn.Linear(input_size, output_size),
            )
            for name, param in self.align.named_parameters():
                if "weight" in name:
                    torch.nn.init.trunc_normal_(param, std=1e-3)
                elif "bias" in name:
                    torch.nn.init.zeros_(param)
        else:
            self.align = None

    def forward(self, vision_features):
        rfc_features = self.rfc(vision_features)
        combined_features = rfc_features + vision_features
        if self.align is not None:
            combined_features = self.align(combined_features)
        return combined_features
    

class PropertyClassifier(nn.Module):
    def __init__(self, input_size):
        super(PropertyClassifier, self).__init__()
        self.act = nn.GELU()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 512),
            self.act,
            nn.Linear(512, 256),
            self.act
        )
        self.hardness_fc = nn.Linear(256, 1)
        self.roughness_fc = nn.Linear(256, 1)

    def forward(self, vision_features):
        vision_features = self.fc(vision_features)
        hardness_preds = self.hardness_fc(vision_features)
        roughness_preds = self.roughness_fc(vision_features)
        preds = torch.cat([hardness_preds, roughness_preds], dim=1)
        return preds
    

def load_encoder(configs, device):
    print("")
    if configs["load_exp_path"] is None:
        load_exp_configs = None
    else:
        load_exp_configs = yaml.safe_load(open(os.path.join(configs["load_exp_path"], "run.yaml"), 'r'))
    if "prompt_learning.yaml" in os.listdir(configs["load_exp_path"]):
        prompt_learning_configs = yaml.safe_load(open(os.path.join(configs["load_exp_path"], "prompt_learning.yaml")))
        clip = PromptLearningCLIPModel.from_pretrained(prompt_learning_configs["use_clip"], prompt_learning_configs).to(device)
    else:
        clip = CLIPModel.from_pretrained(configs["use_clip"]).to(device)
    tactile_vificlip = ViFiCLIP(clip, freeze_text_encoder=True, use_positional_embeds=True).to(device)
    if os.path.exists(os.path.join(configs["load_exp_path"], "tactile_vificlip.pt")):
        state_dict = torch.load(os.path.join(configs["load_exp_path"], "tactile_vificlip.pt"), map_location=device, weights_only=True)
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        tactile_vificlip.load_state_dict(state_dict, strict=False)
        print("Loaded tactile ViFi-CLIP!")
    else:
        warnings.warn("No trained tactile ViFi-CLIP model found!")
    dotted_tactile_adapter = Adapter(input_size=load_exp_configs["dim_context_vision"], output_size=load_exp_configs["dim_context_vision"], residual_ratio=load_exp_configs["residual_ratio"]).to(device)
    if os.path.exists(os.path.join(configs["load_exp_path"], "dotted_tactile_adapter.pt")):
        state_dict = torch.load(os.path.join(configs["load_exp_path"], "dotted_tactile_adapter.pt"), map_location=device, weights_only=True)
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        dotted_tactile_adapter.load_state_dict(state_dict, strict=True)
        print("Loaded dotted tactile adapter!")
    else:
        warnings.warn("No trained dotted tactile adapter found!")
    plain_tactile_adapter = Adapter(input_size=load_exp_configs["dim_context_vision"], output_size=load_exp_configs["dim_context_vision"], residual_ratio=load_exp_configs["residual_ratio"]).to(device)
    if os.path.exists(os.path.join(configs["load_exp_path"], "plain_tactile_adapter.pt")):
        state_dict = torch.load(os.path.join(configs["load_exp_path"], "plain_tactile_adapter.pt"), map_location=device, weights_only=True)
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        plain_tactile_adapter.load_state_dict(state_dict, strict=True)
        print("Loaded plain tactile adapter!")
    else:
        warnings.warn("No trained plain tactile adapter found!")
    property_classifier = PropertyClassifier(input_size=load_exp_configs["dim_context_vision"]).to(device)
    if os.path.exists(os.path.join(configs["load_exp_path"], "property_classifier.pt")):
        state_dict = torch.load(os.path.join(configs["load_exp_path"], "property_classifier.pt"), map_location=device, weights_only=True)
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        property_classifier.load_state_dict(state_dict, strict=True)
        print("Loaded property regression model!")
    else:
        warnings.warn("No trained property regression model found!")
    print("")
    return tactile_vificlip, dotted_tactile_adapter, plain_tactile_adapter, property_classifier, load_exp_configs


def generate_rag_embeddings(configs, load_exp_configs, tactile_vificlip, device, sample_dir, embedding_dir, datasets=["physiclear"], splits=["train"]):
    tactile_vificlip.eval()

    # Get sample count
    sample_count = 0
    for sample in os.listdir(sample_dir):
        dataset = sample.split("_")[0]
        if dataset not in datasets:
            continue
        else:
            sample_path = os.path.join(sample_dir, sample)
            data = json.load(open(os.path.join(sample_path, "data.json"), "r"))
            if data["split"] not in splits:
                continue
            sample_count += 1

    saved_count = 0
    os.makedirs(embedding_dir, exist_ok=True)
    for sample in os.listdir(sample_dir):
        # Save embeddings only for relevant training samples
        dataset = sample.split("_")[0]
        if dataset not in datasets:
            continue
        sample_path = os.path.join(sample_dir, sample)
        data = json.load(open(os.path.join(sample_path, "data.json"), "r"))
        if data["split"] not in splits:
            continue
        if "object" not in data.keys(): # Make sure sample has semantic object before adding to RAG embeddings
            continue
        embedding_path = os.path.join(embedding_dir, f"{sample}.pt")
        image_transforms = get_image_transforms(load_exp_configs["frame_size"], dataset, split_name="test", flip_p=load_exp_configs["flip_p"])
        tactile = os.path.join(sample_path, "tactile")
        tactile_frames, _ = get_frames(tactile, None, image_transforms, frame_size=load_exp_configs["frame_size"], train=False, return_indices=True)
        tactile_frames = torch.unsqueeze(tactile_frames, dim=0)
        tactile_video_features, _, _, _ = tactile_vificlip(tactile_frames.to(device), None, None)
        torch.save(torch.squeeze(tactile_video_features.cpu(), dim=0), embedding_path)
        saved_count += 1
        if saved_count % 100 == 0:
            print(f"Generated {saved_count} / {sample_count} RAG embeddings.")
    print("Done!")
    

def get_rag_embeddings(configs, device):
    object_ids = []
    sample_tactile_paths = []
    saved_embeddings = []
    for embedding in os.listdir(configs["embedding_dir"]):
        sample_path = os.path.join(configs["rag_sample_dir"], embedding.split(".")[0])
        sample_tactile_paths.append(os.path.join(sample_path, "tactile"))
        data = json.load(open(os.path.join(sample_path, "data.json"), "r"))
        object_id = data["object_id"]
        object_ids.append(object_id)
        saved_embeddings.append(torch.load(os.path.join(configs["embedding_dir"], embedding), weights_only=True))
    saved_embeddings = torch.stack(saved_embeddings).to(device)
    return saved_embeddings, sample_tactile_paths, object_ids