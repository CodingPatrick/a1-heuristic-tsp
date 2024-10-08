import random
import numpy as np
import csv
import sys
import matplotlib.pyplot as plt

# Function to read the TSP file and return a dictionary with the city id as key and the coordinates as value
def read_tsp(x):
    city_list = {}
    with open(x, 'r') as file:
        lines = file.readlines()
        flag = False
        for line in lines:
            if line.startswith("NODE_COORD_SECTION"):
                flag = True
                continue
            elif line.startswith("EOF"):
                break
            if flag:
                temp = line.strip().split()
                city_id = int(temp[0])
                x = float(temp[1])
                y = float(temp[2])
                city_list[city_id] = (x, y)
    return city_list

# Function to calculate the Euclidean distance between two points
def euclidean_distance(n1, n2):
    return np.sqrt((n1[0] - n2[0])**2 + (n1[1] - n2[1])**2)

# Function to calculate the distance matrix between all the cities
def distance_matrix(coords_list):
    matrix = np.zeros((len(coords_list), len(coords_list)))
    for i in range(len(coords_list)):
        for j in range(i+1, len(coords_list)):
            distance = euclidean_distance(coords_list[i], coords_list[j])
            matrix[i][j] = distance
            matrix[j][i] = distance 
    return matrix

# Function to generate a random solution for the TSP
def random_solution(nodes_list):
    cities = list(range(1, len(nodes_list) + 1))
    random.shuffle(cities) 
    return cities

# Function to calculate the length of the route
def route_length(current_solution, distance_matrix):
    route_length = 0
    for i in range(len(current_solution) - 1):
        route_length += distance_matrix[current_solution[i] - 1][current_solution[i + 1] - 1] 
    route_length += distance_matrix[current_solution[-1] - 1][current_solution[0] - 1]
    return route_length

# Function to get the neighbors of the current solution
def get_neighbors(current_solution):
    neighbors = []
    for i in range(len(current_solution)):
        for j in range(i+1, len(current_solution)):
            neighbor = current_solution.copy()
            neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
            neighbors.append(neighbor)
    return neighbors

# Function to get the best neighbor
def get_best_neighbor(neighbors, distance_matrix):
    best_neighbor = neighbors[0]
    best_route_length = route_length(best_neighbor, distance_matrix)
    for neighbor in neighbors:
        current_route_length = route_length(neighbor, distance_matrix)
        if current_route_length < best_route_length:
            best_route_length = current_route_length
            best_neighbor = neighbor
    return best_neighbor, best_route_length

# Function to run the Hill Climbing Algorithm with random restart for the TSP
def hill_climbing(city_list, distance_matrix, iterations=1000, restart_threshold=10):

    best_solution = None
    best_route_lengths = []
    best_route_length = float('inf')
    no_improvement_count = 0
    
    for _ in range(iterations):
        
        if no_improvement_count >= restart_threshold:
            # Random restart after the threshold
            current_solution = random_solution(city_list)
            current_route_length = route_length(current_solution, distance_matrix)
            no_improvement_count = 0  # Reset the no improvement count after restart
        else:
            current_solution = random_solution(city_list)
            current_route_length = route_length(current_solution, distance_matrix)
        
        neighbors = get_neighbors(current_solution)
        best_neighbor, best_neighbor_length = get_best_neighbor(neighbors, distance_matrix)
        flag = True
        while flag:
            if best_neighbor_length < current_route_length:
                current_route_length = best_neighbor_length
                current_solution = best_neighbor
                neighbors = get_neighbors(current_solution)
                best_neighbor, best_neighbor_length = get_best_neighbor(neighbors, distance_matrix)
                no_improvement_count = 0  # Reset the no improvement count
            else:
                flag = False
                no_improvement_count += 1  # Increment no improvement count

        if current_route_length < best_route_length:
            best_route_length = current_route_length
            best_solution = current_solution

        best_route_lengths.append(best_route_length)

    return best_solution, best_route_length, best_route_lengths

# Main function to run the Hill Climbing Algorithm with random restart for the TSP
def main():

    # Check if the user has provided the correct number of arguments
    if len(sys.argv) != 2:
        print("Usage: python tsp.py <filename>")
        return
    
    # Read the TSP file and get the city dictionary
    tsp_file = sys.argv[1]
    city_list = read_tsp(tsp_file)

    # Get the coordinates list and the distance matrix
    coords_list = [city_list[city] for city in sorted(city_list)]
    dist_matrix  = distance_matrix(coords_list)

    # Run the Hill Climbing Algorithm with random restart
    best_solution, best_route_length, best_route_lengths = hill_climbing(city_list, dist_matrix)
    print(f"Best route length: {best_route_length}")

    # Write the solution to a csv file
    with open("solution.csv", mode='w', newline='') as file:
        writer = csv.writer(file)
        for city in best_solution:
            writer.writerow([city])

    # Plot the improvement of the Hill Climbing Algorithm after each iteration
    plt.plot(range(0, len(best_route_lengths)), best_route_lengths)
    plt.xlabel('Iteration')
    plt.ylabel('Best Route Length')
    plt.title('Improvement of the Hill Climbing Algorithm with Random Restart after each Iteration for the TSP')
    plt.show()

# Run the main function
if __name__ == "__main__":
    main()
