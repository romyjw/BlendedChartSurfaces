'''
Script to generate a polyline, stored as an obj file, of a silhouette-type image.
Can be used as input for the 2d sdf fitter, which makes approximate sdfs that can be used to demonstrate the curve-fitter.
'''


import cv2
import numpy as np
import sys
sys.path.append('..')

def fit_in_bbox(vertices):
    
    min_vals = np.min(vertices, axis=0)
    max_vals = np.max(vertices, axis=0)
    
    # Shift the vertices by subtracting the minimum values for each axis
    vertices = vertices - min_vals

    ranges = max_vals - min_vals
    scale = 1.0 / np.max(ranges)
    print(scale)
    
    # Scale the shifted vertices so that the largest extent in any axis is 1
    vertices = vertices * scale
    
    vertices = 2 * vertices - 1

    new_min_vals = np.min(vertices, axis=0)
    new_max_vals = np.max(vertices, axis=0)

    vertices = vertices - (new_min_vals+new_max_vals)/2
    
    return vertices

    
def reparametrize_boundary(vertices, num_points):
    """
    Reparametrize the boundary to have equally spaced vertices.

    Parameters:
        vertices (list of tuple): List of (x, y) vertices.
        num_points (int): Desired number of equally spaced vertices.

    Returns:
        list of tuple: Reparametrized vertices.
    """
    vertices = np.array(vertices, dtype=np.float64)
    distances = np.sqrt(np.sum(np.diff(vertices, axis=0, append=vertices[:1])**2, axis=1))
    cumulative_distances = np.cumsum(distances)
    cumulative_distances = np.insert(cumulative_distances, 0, 0)
    total_length = cumulative_distances[-1]

    new_distances = np.linspace(0, total_length, num_points)
    reparametrized_vertices = []

    for d in new_distances:
        idx = np.searchsorted(cumulative_distances, d, side='right') - 1
        t = (d - cumulative_distances[idx]) / distances[idx]
        new_vertex = (1 - t) * vertices[idx] + t * vertices[(idx + 1) % len(vertices)]
        reparametrized_vertices.append(new_vertex)

    return np.array(reparametrized_vertices).tolist()

def subdivide_and_smooth(vertices, iterations=1, smoothing_steps=3, h=0.5):
    """
    Subdivides and smooths the boundary using Laplacian smoothing.

    Parameters:
        vertices (list of tuple): List of (x, y) vertices.
        iterations (int): Number of subdivision steps.
        smoothing_steps (int): Number of Laplacian smoothing steps.

    Returns:
        list of tuple: Smoothed vertices.
    """
    vertices = np.array(vertices, dtype=np.float64)

    for iteration in range(iterations):
        # Subdivide: Insert midpoints between consecutive vertices
        new_vertices = []
        for i in range(len(vertices)):
            p1 = vertices[i]
            p2 = vertices[(i + 1) % len(vertices)]
            midpoint = (p1 + p2) / 2
            new_vertices.append(p1)
            new_vertices.append(midpoint)
        vertices = np.array(new_vertices)

        # Visualize after subdivision
        subdiv_image = np.zeros((500, 500, 3), dtype=np.uint8)
        for i in range(len(vertices)):
            x1, y1 = vertices[i]
            x2, y2 = vertices[(i + 1) % len(vertices)]
            cv2.line(subdiv_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 1)
        cv2.imshow(f"Subdivision Iteration {iteration + 1}", subdiv_image)
        cv2.waitKey(0)

    for step in range(smoothing_steps):
        smoothed_vertices = np.copy(vertices)
        for i in range(len(vertices)):
            prev_idx = (i - 1) % len(vertices)
            next_idx = (i + 1) % len(vertices)
            smoothed_vertices[i] = (1-h)*vertices[i] + h*(vertices[prev_idx] + vertices[next_idx]) / 2
        vertices = smoothed_vertices

        # Visualize after smoothing
        smooth_image = np.zeros((500, 500, 3), dtype=np.uint8)
        for i in range(len(vertices)):
            x1, y1 = vertices[i]
            x2, y2 = vertices[(i + 1) % len(vertices)]
            cv2.line(smooth_image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 1)
        cv2.imshow(f"Smoothing Step {step + 1}", smooth_image)
        cv2.waitKey(0)

    return vertices.tolist()

def extract_boundary_and_save_obj(image_path, obj_path):
    # Load the black-and-white image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"The file {image_path} could not be loaded.")

    # Threshold the image to ensure it's binary (0 and 255)
    _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    
    binary = 255 - binary # Flip black and white
    # Visualize the binary image
    cv2.imshow("Binary Image", binary)
    cv2.waitKey(0)

    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Visualize all contours
    contour_image = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)
    cv2.imshow("Contours", contour_image)
    cv2.waitKey(0)

    # Extract the top 3 largest contours (by area)
    if not contours:
        raise ValueError("No contours found in the image.")

    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)[:3]

    # Visualize the top 3 contours
    top_contours_image = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # Colors for the top 3 contours
    for i, contour in enumerate(sorted_contours):
        cv2.drawContours(top_contours_image, [contour], -1, colors[i % len(colors)], 2)
    cv2.imshow("Top 3 Contours", top_contours_image)
    cv2.waitKey(0)

    # Use the largest contour for OBJ file
    largest_contour = sorted_contours[0]

    # Prepare vertices and edges for the OBJ file
    vertices = [(point[0][0], point[0][1]) for point in largest_contour]

    # Reparametrize to equally spaced vertices
    #vertices = reparametrize_boundary(vertices, num_points=100)

    # Subdivide and smooth the vertices
    vertices = subdivide_and_smooth(vertices, iterations=0, smoothing_steps=10)

    edges = [(i + 1, (i + 2) if (i + 1) < len(vertices) else 1) for i in range(len(vertices))]
    
    
    
    
    #normalize
    vertices = np.array(vertices)
    vertices = fit_in_bbox(vertices)
    
    vertices[:,1] = -1*vertices[:,1]
    
    
    
    
    
    # Write to the OBJ file
    with open(obj_path, 'w') as obj_file:
        obj_file.write("# OBJ file generated from image boundary\n")

        # Write vertices
        for x, y in vertices:
            obj_file.write(f"v {x} {y} 0\n")

        # Write edges
        for start, end in edges:
            obj_file.write(f"l {start} {end}\n")

    print(f"OBJ file saved to {obj_path}")

# Example usage
name = sys.argv[-1]

image_path = "data/images/"+name+".png"  # Replace with your image path
obj_path = "data/curves/"+name+"_bdry.obj"  # Replace with your desired OBJ file path

extract_boundary_and_save_obj(image_path, obj_path)
