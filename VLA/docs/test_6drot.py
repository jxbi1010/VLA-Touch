import numpy as np
from scipy.spatial.transform import Rotation as R


def convert_quaternion_to_euler(quat):
    """
    Convert Quarternion (xyzw) to Euler angles (rpy) 
    """
    # Normalize
    quat = quat / np.linalg.norm(quat)
    euler = R.from_quat(quat).as_euler('xyz')

    return euler


def convert_euler_to_quaternion(euler):
    """
    Convert Euler angles (rpy) to Quarternion (xyzw)
    """
    quat = R.from_euler('xyz', euler).as_quat()
    
    return quat


def convert_euler_to_rotation_matrix(euler):
    """
    Convert Euler angles (rpy) to rotation matrix (3x3).
    """
    quat = R.from_euler('xyz', euler).as_matrix()
    
    return quat


def convert_rotation_matrix_to_euler(rotmat):
    """
    Convert rotation matrix (3x3) to Euler angles (rpy).
    """
    r = R.from_matrix(rotmat)
    euler = r.as_euler('xyz', degrees=False)
    
    return euler


def normalize_vector(v):
    v_mag = np.linalg.norm(v, axis=-1, keepdims=True)
    v_mag = np.maximum(v_mag, 1e-8)
    return v / v_mag


def cross_product(u, v):
    i = u[:,1]*v[:,2] - u[:,2]*v[:,1]
    j = u[:,2]*v[:,0] - u[:,0]*v[:,2]
    k = u[:,0]*v[:,1] - u[:,1]*v[:,0]
        
    out = np.stack((i, j, k), axis=1)
    return out
        

def compute_rotation_matrix_from_ortho6d(ortho6d):
    x_raw = ortho6d[:, 0:3]
    y_raw = ortho6d[:, 3:6]
        
    x = normalize_vector(x_raw)
    z = cross_product(x, y_raw)
    z = normalize_vector(z)
    y = cross_product(z, x)
    
    x = x.reshape(-1, 3, 1)
    y = y.reshape(-1, 3, 1)
    z = z.reshape(-1, 3, 1)
    matrix = np.concatenate((x, y, z), axis=2)
    return matrix


def compute_ortho6d_from_rotation_matrix(matrix):
    # The ortho6d represents the first two column vectors a1 and a2 of the
    # rotation matrix: [ | , |,  | ]
    #                  [ a1, a2, a3]
    #                  [ | , |,  | ]
    ortho6d = matrix[:, :, :2].transpose(0, 2, 1).reshape(matrix.shape[0], -1)
    return ortho6d


def generate_random_quaternion():
    """
    Generates a random unit quaternion (normalized) using NumPy.

    Returns:
        numpy.ndarray: A quaternion as a NumPy array [w, x, y, z]
    """
    # Generate 3 random values between 0 and 1
    u1 = np.random.random()
    u2 = np.random.random()
    u3 = np.random.random()

    # Square root of random value for proper distribution
    sqrt1_minus_u1 = np.sqrt(1 - u1)
    sqrt_u1 = np.sqrt(u1)

    # Calculate quaternion components using the random values
    w = sqrt1_minus_u1 * np.sin(2 * np.pi * u2)
    x = sqrt1_minus_u1 * np.cos(2 * np.pi * u2)
    y = sqrt_u1 * np.sin(2 * np.pi * u3)
    z = sqrt_u1 * np.cos(2 * np.pi * u3)

    # Return the quaternion as a NumPy array
    return np.array([w, x, y, z])


def convert_quaternion_to_orthod6d(quat):

    euler = convert_quaternion_to_euler(quat)
    rotmat = convert_euler_to_rotation_matrix(euler)
    ortho6d = compute_ortho6d_from_rotation_matrix(rotmat)

    return ortho6d

# Test
if __name__ == "__main__":
    # # Randomly generate a euler ange
    # euler = np.random.rand(3) * 2 * np.pi - np.pi
    # euler = euler[None, :]    # Add batch dimension
    # print(f"Input Euler angles: {euler}")
    #
    # # Convert to 6D Rotation
    # rotmat = convert_euler_to_rotation_matrix(euler)
    # ortho6d = compute_ortho6d_from_rotation_matrix(rotmat)
    # print(f"6D Rotation: {ortho6d}")
    #
    # # Convert back to Euler angles
    # rotmat_recovered = compute_rotation_matrix_from_ortho6d(ortho6d)
    # euler_recovered = convert_rotation_matrix_to_euler(rotmat_recovered)
    # print(f"Recovered Euler angles: {euler_recovered}")


    quoternion = generate_random_quaternion()
    quoternion = quoternion[None,:]
    print(f"Input quoternion: {quoternion}")
    euler = convert_quaternion_to_euler(quoternion)
    # euler = euler[None, :]
    print(f"Input Euler angles: {euler}")
    rotmat = convert_euler_to_rotation_matrix(euler)
    ortho6d = compute_ortho6d_from_rotation_matrix(rotmat)
    print(f"6D Rotation: {ortho6d}")
    rotmat_recovered = compute_rotation_matrix_from_ortho6d(ortho6d)
    euler_recovered = convert_rotation_matrix_to_euler(rotmat_recovered)
    print(f"Recovered Euler angles: {euler_recovered}")
    quoternion_recovered =convert_euler_to_quaternion(euler_recovered)
    print(f"Recovered quoternion_recovered: {quoternion_recovered[0]}")
