import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def ccw(A, B, C):
    #Check order of points- anti-clockwise
    return (C[1]-A[1])*(B[0]-A[0]) > (B[1]-A[1])*(C[0]-A[0])

def intersect(A, B, C, D):
    #Segments Ab and CD intersect if ABC and ABD have opposite sense AND ACD and BCD have opposite sense
    return (ccw(A,C,D) != ccw(B,C,D)) and (ccw(A,B,C) != ccw(A,B,D))

def point_in_poly(point, poly):
    #Returns true of one of the vertices of the graph lies in the interior of one of the obstacles and is inaccessible
    x, y = point
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i+1) % n] #Assuming that vertices of polygons are labeled consecutively- the next index is of the vertex that forms an edge of the polygon
        if ((y1 > y) != (y2 > y)) and \
           (x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-12) + x1): #Intersection of a horizontal line from the point with the edge of the polygon
            inside = not inside
    #If the horizontal line intersects the edges an even number of times, the point is ouside
    #If it intersects an odd number of times, the point is inside the polygon
    return inside

def construct_visibility_graph(raw_obstacles, start, goal,bounds):
    Graph = nx.Graph()
    n_vertices = raw_obstacles.shape[0]
    xmin,xmax,ymin,ymax=bounds
    # Add nodes for all obstacle vertices
    Graph.add_nodes_from(range(n_vertices))
    for node in range(n_vertices):
        Graph.nodes[node]['coordinates'] = raw_obstacles[node, 0:2]

    #Remove vertices tht are either inside another obstacle or are outside of the bounds of the environment
    #This is necessary as the "obstacles" fed to the algorithm are expanded by 0.5 the width of the thymio
    vertex_blocked = np.zeros(n_vertices, dtype=bool)
    for v in range(n_vertices):
        px, py = raw_obstacles[v,0], raw_obstacles[v,1]
        if (px>=xmax or py>=ymax or px<=xmin or py<=ymin):
            vertex_blocked[v] = True
            # print('Out of bounds ',v)
    # Add start and goal nodes
    Graph.add_node('S', coordinates=np.array(start))
    Graph.add_node('G', coordinates=np.array(goal))

    # Organize obstacles by their index
    obstacles = []
    for i in range(int(max(raw_obstacles[:,2])+1)):
        vertices = raw_obstacles[raw_obstacles[:,2]==i, 0:2]
        obstacles.append(vertices)
    for v in range(n_vertices):
        p = raw_obstacles[v,0:2]
        own_id = raw_obstacles[v,2]
        for obs_id in range(len(obstacles)):
            # skip its own polygon
            if np.all(obs_id==own_id):
                continue
            obs=obstacles[obs_id]
            if point_in_poly(p, obs):
                vertex_blocked[v] = True
                break
    # Helper to check if a candidate edge intersects any obstacle
    def is_edge_free(p1, p2, obstacle_list):
        for poly in obstacle_list:
            n = len(poly)
            for i in range(n):
                q1 = poly[i]
                q2 = poly[(i+1)%n]
                # Avoid checking the same vertex (adjacent polygon edge)
                # This means obstacle diagonals are excluded, but this is handled outside of the helper function
                if np.all(p1==q1) or np.all(p1==q2) or np.all(p2==q1) or np.all(p2==q2):
                    continue
                if intersect(p1, p2, q1, q2):
                    return False
        return True

    # Add edges between obstacle vertices
    for i in range(n_vertices-1):
        if vertex_blocked[i]:
            #Exclude out of bounds and obstacle ioverlapping vertices
            continue
        for j in range(i+1, n_vertices):
            if vertex_blocked[j]:
                continue
            p1 = raw_obstacles[i,0:2]
            p2 = raw_obstacles[j,0:2]
            if raw_obstacles[i,2] == raw_obstacles[j,2]:
                poly_indices = np.where(raw_obstacles[:,2]==raw_obstacles[i,2])[0]
                if not(j == i+1 or (i==poly_indices[0] and j==poly_indices[-1])):
                    #Exclude obstacle digonals as they are not valid edges
                    continue
            if is_edge_free(p1, p2, obstacles):
                #If the edge is valid, add it to the graph with the weigt as Euclidean distance
                dist = np.linalg.norm(p1-p2)
                Graph.add_edge(i,j,weight=dist)

    # Add edges from start and goal to obstacle vertices
    for v in range(n_vertices):
        if vertex_blocked[v]:
            continue
        p = raw_obstacles[v,0:2]
        if is_edge_free(start, p, obstacles):
            Graph.add_edge('S', v, weight=np.linalg.norm(start-p))
        if is_edge_free(goal, p, obstacles):
            Graph.add_edge('G', v, weight=np.linalg.norm(goal-p))

    # Add direct edge between start and goal if free
    if is_edge_free(start, goal, obstacles):
        Graph.add_edge('S', 'G', weight=np.linalg.norm(start-goal))

    return Graph

def plot_visibility_graph(Graph, raw_obstacles, start, goal, path=None, arrow_scale=5):
    plt.figure(figsize=(8,6))

    # Plot obstacles as filled polygons
    obstacle_ids = np.unique(raw_obstacles[:,2])
    for obs_id in obstacle_ids:
        vertices = raw_obstacles[raw_obstacles[:,2]==obs_id, 0:2]
        poly = np.vstack((vertices, vertices[0]))  # close the polygon
        plt.fill(poly[:,0], poly[:,1], color='red', alpha=0.6)

    # Node positions for nx.draw extracted from coordinates
    pos = {node: attr['coordinates'] for node, attr in Graph.nodes(data=True)}

    nx.draw(Graph, pos, with_labels=True, node_color='skyblue', node_size=100, font_size=10, edge_color='k')
    nx.draw_networkx_nodes(Graph, pos, nodelist=['S'], node_color='green', node_size=150)
    nx.draw_networkx_nodes(Graph, pos, nodelist=['G'], node_color='blue', node_size=150)

    # If a path is provided, plot it with arrows
    if path is not None and len(path) > 0:
        path_coords = np.array([(x, y) for x, y, _ in path])
        plt.plot(path_coords[:,0], path_coords[:,1], color='orange', linewidth=2, marker='o', label='Path')

        # Plot heading arrows using quiver
        for x, y, theta in path:
            dx = arrow_scale * np.cos(theta)
            dy = arrow_scale * np.sin(theta)
            plt.plot([x,x+dx], [y,y+dy], color='blue', linewidth=2, label='Path')
            plt.plot([x+dx], [y+dy], color='blue', linewidth=2, marker='o', label='Path')

    plt.axis('equal')
    plt.xlabel('X')
    plt.ylabel('Y')
    if path is not None:
        plt.title('Visibility Graph with obstacles, path and heading angle')
    else:
        plt.title('Visibility Graph with obstacles')
    plt.show()


def plan_global_navigation(G,start,goal):
    #Dijkstra algorithm used to compute the optimal path from start to goal on the given visibility graph
    path =nx.dijkstra_path(G,start,goal)
    return path

def compute_orientations(G,path):
    #With given path, compute target heading angle of the thymio for each vertex visited
    theta=[]
    for i in range(len(path)-1):
        end_node=G.nodes[path[i+1]]['coordinates']
        start_node=G.nodes[path[i]]['coordinates']
        if abs(end_node[0]-start_node[0])<0.00001:
            #Infinite slope case for vertical line
            angle=np.pi/2
            theta.append(angle)
            print('end node:',end_node,' start node:',start_node)
        else:
            slope_num=(end_node[1]-start_node[1])
            slope_denom=(end_node[0]-start_node[0])
            angle=np.arctan2(slope_num,slope_denom)
            #angle at the start of an edge should be equal to arctan(slope of this edge)
            theta.append(angle)
            print('theta:',angle,' end node:',end_node,' start node:',start_node)
    theta.append(angle)
    #The control strategy followed is first the orientation of the robot is corrected in pure rotation at each vertex
    # and then the robot continues along the edge in a straight line.
    new_path=[]
    for i in range(len(path)):
        xy_data=G.nodes[path[i]]['coordinates']
        new_path.append((xy_data[0],xy_data[1],theta[i]))
    #The path is returned with position_x, position_y and position_theta of the desired trajectory
    return new_path
