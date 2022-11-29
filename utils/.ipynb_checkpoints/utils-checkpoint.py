import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def reduce_to_manhattan_rectangular(points):
    """
    Reduces hypothetical polygons of more than 4 vertices, to rectangulars if that is true.
    
    Args:
        points (array): A (N,2) numpy array of vertex points.
    Returns:
        points: A list of vertex points, possibly reduces.
    """
    #Get the vertices that enclose the polygon
    min_x, max_x = np.where(points[:,0]==np.min(points[:,0]))[0], np.where(points[:,0]==np.max(points[:,0]))[0]
    min_y, max_y = np.where(points[:,1]==np.min(points[:,1]))[0], np.where(points[:,1]==np.max(points[:,1]))[0]
    if len(min_x) < 2 or len(max_x) < 2 or len(min_y) < 2 or len(max_y) <2:
        return points
    #Return the vertices on the edges that create an rectangular
    return points[np.concatenate((np.intersect1d(min_x, min_y), np.intersect1d(min_x, max_y), np.intersect1d(max_x, max_y), np.intersect1d(max_x, min_y)))]

def order_quadrilateral_vertices(quad):
    """
    Orders clockwise from top left to bottom left the vertices of a quadrilateral.
    
    Args:
        quad (array): a (4,2) numpy array of vertex points.
    Returns:
        points: ordered list of vertex points, (4,2).
    """
    #Sort quadrilateral by x axis.
    xord = quad[np.argsort(quad[:,0])]
    #Keep the leftmost and the rightmost coordinates.
    leftmost, rightmost = xord[:2], xord[2:]
    #Sort topleftmost and toprightmost by y axis.
    topleftmost = leftmost[np.argsort(-leftmost[:,1])]
    toprightmost = rightmost[np.argsort(-rightmost[:,1])]
    #Return the sorted quadrilateral.
    return np.reshape(np.concatenate((topleftmost[0], toprightmost[0], toprightmost[1], topleftmost[1])), (4,2))

def check_if_point_inside_a_rectangular(quad, query):
    """Checks if a query point resides inside a quadrilateral.
    
    Args:
        quad (array): a (4,2) numpy array of vertex points.
        query (array or list): B (2) or (1,2) numpy array or list of a point.
    Returns:
        True or False, if a points resides in the quadrilateral or not, respectively.
    """
    AB, AM, BC, BM = quad[1]-quad[0], query-quad[0], quad[2]-quad[1], query-quad[1]
    if (0 <= np.dot(AB, AM) <= np.dot(AB,AB)) & (0 <= np.dot(BC, BM) <= np.dot(BC,BC)):
        return True
    return False

def slope(a,b):
    """
    Compute the slope between two points in R^2.
    
    Args:
        a (array or list): (2) or (1,2) numpy array or list of a point's coordinates in R^2 space.
        b (array or list): (2) or (1,2) numpy array or list of another point's coordinates in R^2 space.
    Returns:
        The slope of the two points a and b.
    """
    #Compute denominator
    denom = (b[0]-a[0])
    #If denominator is 0
    if not denom:
        #Make it close to 0
        denom = 10e-4
    #Return the slope of the two points
    return (b[1]-a[1])/denom

def check_if_rectangular(quad):
    """
    Check if your quadrilateral is rectangular, thus check if there are two pairs of parallel lines in your quadrilateral.
    
    Args:
        quad (array): a (4,2) numpy array of vertex points.
    Returns:
        True or False, if your quadrilateral is rectangular or not, respectively.
    """
    return slope(quad[0],quad[1]) == slope(quad[3],quad[2]) and slope(quad[3],quad[0]) == slope(quad[2],quad[1])

def position_of_point_towards_line(a, b, q):
    """
    Check the direction of a point in respect with a line consists of two points in R^2.
    The direction of the line-points should be from bottom to top for vertical lines and from right to left for horizontal lines.
    Positive number means that your point exist on the top or right of your line.
    Negative number means that your point exist on the bottom or left of your line.
    Zero indicates that your point is on your line.
    For more: https://stackoverflow.com/questions/1560492/how-to-tell-whether-a-point-is-to-the-right-or-left-side-of-a-line
    
    Args:
        a (array or list): (2) or (1,2) numpy array or list of a point's coordinates in R^2 space.
        b (array or list): (2) or (1,2) numpy array or list of another point's coordinates in R^2 space.
        q (array or list): (2) or (1,2) numpy array or list of a query point's coordinates in R^2 space.
    Returns:
        Float, that its sign and value indicates the direction of a point in respect with a line.
    """
    return (q[0]-a[0])*(b[1]-a[1])-(q[1]-a[1])*(b[0]-a[0])

def check_if_point_inside_conductors(conductors,q):
    """
    Check if a point resides inside the region of the two parallel sets of sides of a rectangular. 
    
    Args:
        conductors (array): a (N,4,2) numpy array of vertex points.
        q (array or list): (2) or (1,2) numpy array or list of a query point's coordinates in R^2 space.
    Returns:
        True if the point is inside the quadrilateral
        False if the point is outside of the quadrilateral
    """
    #Compute the position of the query towards vertical lines of conductors
    left, right = position_of_point_towards_set_of_lines(conductors[:,3],conductors[:,0],q), position_of_point_towards_set_of_lines(conductors[:,2],conductors[:,1],q)
    #Compute the position of the query towards horizontal lines of conductors
    top, bottom = position_of_point_towards_set_of_lines(conductors[:,1],conductors[:,0],q), position_of_point_towards_set_of_lines(conductors[:,2],conductors[:,3],q)
    point_orientation = np.array([left, right, top, bottom])
    #If query point between left and right and top and bottom sides of the quadrilateral conductor.
    if len(np.where((point_orientation[0,:]*point_orientation[1,:]<0)&(point_orientation[2,:]*point_orientation[3,:]<0))[0]):
        return True
    return False

def position_of_point_towards_set_of_lines(a, b, q):
    """
    Check the direction of a point in respect with a line consists of two points in R^2.
    The direction of the line-points should be from bottom to top for vertical lines and from right to left for horizontal lines.
    Positive number means that your point exist on the top or right of your line.
    Negative number means that your point exist on the bottom or left of your line.
    Zero indicates that your point is on your line.
    For more: https://stackoverflow.com/questions/1560492/how-to-tell-whether-a-point-is-to-the-right-or-left-side-of-a-line
    
    Args:
        a (array): (N,2) shape numpy array or list of a points' coordinates in R^2 space.
        
        b (array): (N,2) shape numpy array or list of another points' coordinates in R^2 space.
        
        q (array or list): (2) or (1,2) numpy array or list of a query point's coordinates in R^2 space.
        
    Returns:
        Float, that its sign and value indicates the direction of a point in respect with a line.
    """
    return (q[0]-a[:,0])*(b[:,1]-a[:,1])-(q[1]-a[:,1])*(b[:,0]-a[:,0])

def check_if_point_between_parallels_of_set_of_conductors(conductors,q):
    """
    Check if a point resides inside the region of the two parallel sets of sides of a quadrilateral. 
    
    Args:
        conductors (array): a (N,4,2) shape numpy array of vertex points.
        
        q (array or list): (2) or (1,2) numpy array or list of a query point's coordinates in R^2 space.
        
    Returns:
        point_positions_encoded (array) : (N,3) shape array of encoded orientation of query point with respect to the quadrilaterals.
    """
    #Compute the position of the query towards vertical lines of quadrilateral
    left, right = position_of_point_towards_set_of_lines(conductors[:,3],conductors[:,0],q), position_of_point_towards_set_of_lines(conductors[:,2],conductors[:,1],q)
    #Compute the position of the query towards horizontal lines of quadrilateral
    top, bottom = position_of_point_towards_set_of_lines(conductors[:,1],conductors[:,0],q), position_of_point_towards_set_of_lines(conductors[:,2],conductors[:,3],q)
    #Collect point position to quadrants sides.
    point_positions =  np.array([left, right, top, bottom])
    #Initialize array to keep encoded information about the position of point with respect to the conductor
    point_positions_encoded = np.zeros((len(point_positions[0]),3))
    #Point is between the parallels of the vertical line segments for conductors
    between_verticals = np.where((point_positions[0]*point_positions[1])<0)[0]
    #point is between the parallels of the horizontal line segments for conductors
    between_horizontal = np.where((point_positions[2]*point_positions[3])<0)[0]
    #Keep lenghts of each case
    len_between_vert, len_between_hor = len(between_verticals), len(between_horizontal)
    #Encode the position information
    if len_between_vert:
        point_positions_encoded[between_verticals] = np.array([np.ones((len_between_vert)), point_positions[2][between_verticals], point_positions[3][between_verticals]]).T
    if len(between_horizontal):
        point_positions_encoded[between_horizontal] = np.array([np.zeros((len_between_hor)), point_positions[0][between_horizontal], point_positions[1][between_horizontal]]).T
    #Return the collected information
    return point_positions_encoded

def perpendicular_distance_to_a_line(a,b,q):
    """
    Compute perpendicular distance of a query point q from a line defined from two points a and b.
    For more: https://stackoverflow.com/questions/39840030/distance-between-point-and-a-line-from-two-points
    
    Args:
        a (array or list): (2) or (1,2) numpy array or list of a point's coordinates in R^2 space.
        b (array or list): (2) or (1,2) numpy array or list of another point's coordinates in R^2 space.
        q (array or list): (2) or (1,2) numpy array or list of a query point's coordinates in R^2 space.
    Returns:
        Float, the perpendicular distance of q from line of ab.
    """
    return np.linalg.norm(np.cross(b-a, a-q))/np.linalg.norm(b-a)

def ccw(A,B,C):
    """
    A function that determines if three points ∈ R^2, are listed in a counterclockwise order.
    For more see : https://stackoverflow.com/questions/3838329/how-can-i-check-if-two-segments-intersect
    
    Args:
        A (array or list): a len(2) numpy array or list of a point in R^2.
        
        B (array or list): a len(2) numpy array or list of a point in R^2.
        
        C (array or list): a len(2) numpy array or list of a point in R^2.
        
    Returns:
        True or False
    """
    return (C[1]-A[1])*(B[0]-A[0]) > (B[1]-A[1])*(C[0]-A[0])

def intersect(line,C,D):
    """
    Determine if a line segment (A, B) intersects with another line segment (C,D).
    These intersect if and only if points A and B are separated by segment CD and points C and D are separated by segment AB.
    If points A and B are separated by segment CD then ACD and BCD should have opposite orientation meaning either ACD or BCD is counterclockwise but not both.
    For more see : https://stackoverflow.com/questions/3838329/how-can-i-check-if-two-segments-intersect
    
    Args:
        line (class object): object of LinePoints class.
        
        C (array or list): a len(2) numpy array or list of a point in R^2.
        
        D (array or list): a len(2) numpy array or list of a point in R^2.
        
    Returns:
        True or False
    """
    return (ccw([line.x1,line.y1],C,D) != ccw([line.x2,line.y2],C,D) and
            ccw([line.x1,line.y1], [line.x2,line.y2],C) != ccw([line.x1,line.y1], [line.x2,line.y2],D))

def node_intersection(x, y, w, h, line):
    """
    Determine if a line segment intersects with the node (or quadrant).
    
    Args:
        x (float) : The origin of the quadrilateral quadrant in x axis.

        y (float) : The origin of the quadrilateral quadrant in y axis.

        w (float) : The width of the quadrilateral quadrant (x axis dimension).

        h (float) : The height of the quadrilateral quadrant (y axis dimension).
        
        line (class object): object of LinePoints class.
        
    Returns:
        True or False
    """
    #Check if your line intersect with either of the four sides of your node.
    if intersect(line, [x,y], [x,y+h]):
        return True
    elif intersect(line, [x,y+h], [x+w,y+h]):
        return True
    elif intersect(line, [x+w,y+h], [x+w,y]):
        return True
    elif intersect(line, [x+w,y], [x,y]):
        return True
    #If not, return False.
    return False

def contains_point(x, y, w, h, point):
    """
    Checks if the line is contained in the quadrant.
    
    Parameters
    ----------
    x : float
        The origin of the quadrilateral quadrant in x axis.

    y : float
        The origin of the quadrilateral quadrant in y axis.

    w : float
        The width of the quadrilateral quadrant (x axis dimension).

    h : float
        The height of the quadrilateral quadrant (y axis dimension).

    point : array or list
        A len(2) numpy array or list of a point in R^2.
        
    Returns
    -------
    True or False
    """
    return point[0]>=x and point[0]<=x+w and point[1]>=y and point[1]<=y+h

def contains(x, y, w, h, lines):
    """
    Checks if the line is contained in the quadrant.
    
    Parameters
    ----------
    x : float
        The origin of the quadrilateral quadrant in x axis.

    y : float
        The origin of the quadrilateral quadrant in y axis.

    w : float
        The width of the quadrilateral quadrant (x axis dimension).

    h : float
       The height of the quadrilateral quadrant (y axis dimension).

    lines : list
        List with objects of LinePoints class.
        
    Returns
    -------
    List of the points included in the quadrant.
    """
    
    included = []
    #For each line in the set of lines
    for line in lines:
        #If either of the two points of the line segment is inside the quadrant, keep the line.
        if contains_point(x, y, w, h, [line.x1,line.y1]) or contains_point(x, y, w, h, [line.x2,line.y2]):
            included.append(line)
        #If the line intersects the node, keep also the line in the list.
        elif node_intersection(x, y, w, h, line):
            included.append(line)
    return included

def recursive_subdivision(node, k):
    """
    Recursive subdivision of quadrants based on the required threshold k.
    
    Parameters
    ----------
    node : class object
        Class object of parental node (quadrant). For more info see the description of Quadrant class.

    k : int
        Threshold for the number of lines that can be found inside a quadrant.
        
    Returns
    -------
    The complete quadtree structure, with the construction of all the quadrants.
    """
    #If the quadrant has less lines than the threshold return
    if len(node.lines)<=k:
        return
    #New quadrant will have half the width and height of the parental.
    w_ = float(node.width/2)
    h_ = float(node.height/2)
    
    #Create the bottom left quadrant and its sub-quadrants, until the threshold is met.
    p = contains(node.x0, node.y0, w_, h_, node.lines)
    x1 = Quadrant(node.x0, node.y0, w_, h_, p)
    recursive_subdivision(x1, k)
    
    #Create the top left quadrant and its sub-quadrants, until the threshold is met.
    p = contains(node.x0, node.y0+h_, w_, h_, node.lines)
    x2 = Quadrant(node.x0, node.y0+h_, w_, h_, p)
    recursive_subdivision(x2, k)
    
    #Create the bottom right quadrant and its sub-quadrants, until the threshold is met.
    p = contains(node.x0+w_, node.y0, w_, h_, node.lines)
    x3 = Quadrant(node.x0 + w_, node.y0, w_, h_, p)
    recursive_subdivision(x3, k)
    
    #Create the top right quadrant and its sub-quadrants, until the threshold is met.
    p = contains(node.x0+w_, node.y0+h_, w_, h_, node.lines)
    x4 = Quadrant(node.x0+w_, node.y0+h_, w_, h_, p)
    recursive_subdivision(x4, k)
    
    #When finished, keep the children of the node.
    node.children = [x1, x2, x3, x4]

def find_children(node):
    """
    Find descendants of the given node.
    
    Parameters
    ----------
    node : class object
        Class object of parental node (quadrant). For more info see the description of Quadrant class.

    Returns
    -------
    A list with all the "children" of the given node.
    """
    #If the node does note have any children, return a list with only the node in it.
    if not node.children:
        return [node]
    #In any other case, create a list and append all the children of the node, while add to that all the children of the children consequently.
    else:
        children = []
        for child in node.children:
            children += (find_children(child))
    return children

def find_quadrant(quad_info,p):
    """
    Search quadtree starting from root quadrant and going all the way towards the smallest resolution, thus the smallest possible quadrant.
    
    Args:
        quad_info (list or array) : a (M,2) shape numpy array or list of all non empty quadrants.
        
        p (array or list) : a len(2) shape numpy array or list of random query point.
        
    Returns:
        ind (int) : index of the selected quadrant of the quadtree.
    """
    #descendants is basically a True or False variable, if there are no other descendants,
    #then quad_info[0][6] = -1, consequently descendants = 0 and while loop breaks 
    descendants = quad_info[0][6]+1
    #Assign as first node the root of the quadtree
    node = quad_info[0]
    #While there are descendants
    while descendants:
        #Find the direct descendants
        for ind in node[6:]:
            #When you find the one that contains your point break
            if contains_point(*quad_info[ind][:4],p):
                break
        #Assign to descendants variable if there are or no descendants of your new node.
        descendants = quad_info[ind][6]+1
        #And move on to your new node
        node = quad_info[ind]
    #If there are no any more descendants return the index of the smallest quadrant that contain your point.
    return ind

def quadrants_relationships(qtree_root):
    """
    Find all the relationships between direct descendants and ancestors, while defining indexes for the quadrants in every resolution level.
    
    Args:
        qtree_root (class object) : Root quadrant, of class Quadrants
                
    Returns:
        quadrants (list) : a len(M) list with quadrant objects of class Quadrant.
        
        quadrants_info (list) : a (M,10) shape list of all non empty quadrants.
    """
    #Set the first node as your root rectangle, with no node parents.
    nodes, nodes_parent = [qtree_root], []
    #Initilize quadrants lists
    quadrants, quadrants_info = [], []
    #quad indexes
    q_ind = 0
    #While you have nodes to register
    while nodes:
        #Keep the next batch of nodes to register
        new_nodes = []
        #Keep the parents indexes of the next batch of nodes to register
        new_nodes_parent = []
        #For quadrant of the node
        for node_number, quad in enumerate(nodes):
            #Register quadrant in the quadrants list
            quadrants.append(quad)
            #Info contains, quadrant info, index of parent node, current index, initialize children indexes
            quadrants_info.append(quad.get_info()+[-1, q_ind,-1,-1,-1,-1])
            #If there are parental nodes (in every case except root node)
            if len(nodes_parent):
                #Keep track of the ancestor and descendants of the parent
                quadrants_info[-1][4] = nodes_parent[node_number][0]
                quadrants_info[nodes_parent[node_number][0]][nodes_parent[node_number][1]] = q_ind
            #If the quadrant has children
            if quad.children:
                #For every of the 4 children of the parental quadrant
                for child_number, child in enumerate(quad.children):
                    #If the child contains lines keep it. ### THINK IF THIS IS CORRECT ###
                    #if len(child.lines):
                    #Append the child to the next batch of nodes
                    new_nodes.append(child)
                    #And keep the parental index and the number of the child
                    new_nodes_parent.append([q_ind, 6+child_number])
            #At the end move up by one your q indexes
            q_ind += 1
        #Allocate next nodes and parents of next nodes
        nodes = new_nodes
        nodes_parent = new_nodes_parent
    return quadrants, quadrants_info

def random_query_points(ordered_vertices, quadrants_info, ordered_conductors, n_points):
    """
    Create n random query points, which are not inside any conductor.
    
    Args:
        quad (array): a (4,2) numpy array of vertex points.
        
        ordered_vertices (array) : a (N*4,2) shape numpy array of all ordered vertices of the dataset.
        
        quadrants_info (list or array) : a (M,2) shape numpy array or list of all non empty quadrants.
        
        ordered_conductors (array) : a (N,2) shape numpy array of all conductors with ordered quadrilateral of the dataset.
        
        n_points (int) : number of query points to spawn.
        
    Returns:
        random_points (list) : a (n_points,2) shape list of random query points.
    """
    #Initialize list random_points in which you will keep your random query points.
    random_points = []
    #Find xmin, xmax, ymin, ymax from your dataset points.
    xmin, xmax, ymin, ymax = np.min(ordered_vertices[:,0]), np.max(ordered_vertices[:,0]), np.min(ordered_vertices[:,1]), np.max(ordered_vertices[:,1])
    #While you have not find the selected number "n_points" of random query points, keep in the following loop.
    while len(random_points)-n_points:
        #inside variable dictates if your point is inside a conductor or not
        inside = False
        #Find a random point, accordinate to your coordinate minimum and maximum
        random_point = [xmin+(xmax-xmin)*random.random(), ymin+(ymax-ymin)*random.random()]
        #Find the quadrant that your query point resides in, with information from your quadtree
        quadrant_ind = find_quadrant(quadrants_info,random_point)
        #Keep the conductor indexes that are inside your quadrant that your point was found.
        conductors_ind = quadrants[quadrant_ind].unique_conductors_indexes
        #For every conductor check if query point falls into a conductor
        if check_if_point_inside_conductors(ordered_conductors[conductors_ind], random_point):
            continue
        #Append your random point to your random_points list
        random_points.append(random_point)
        print(f'Have been created {len(random_points)}/{n_points} random query points.', end='\r')
    #When out of the while loop return the list with random points
    return random_points

def get_quadrants_vertices(quadrants_info):
    """
    Get the vertices of the quadrants
    
    Args:
        quadrants_info (list or array) : a (M,10) shape list or array of all non empty quadrants.
                
    Returns:        
        quadrant_vertices (array) : a (M,4,2) shape array of all vertices of quadrants.
    """
    #Initialize quadrant_vertices array.
    quadrant_vertices = np.zeros((len(quadrants_info),4,2))
    #For every quadrant
    for i, quadrant in enumerate(quadrants_info):
        #Keep info about x0, y0, width and height
        x,y,w,h = quadrant[:4]
        #Translate this info to vertex coordinates
        quadrant_vertices[i] = [[x,y+h],[x+w,y+h],[x+w,y],[x,y]]
    #Return quadrant vertices
    return quadrant_vertices

def get_quadrant_borders(indexes_of_infertile_quadrants, quadrants_info, quadrant_vertices):
    """
    Compute borders of all infertile quadrants in your quadtree.
    
    Args:
        indexes_of_infertile_quadrants (array) : a len(N) array with indexes of quadrants with no descendants.
        
        quadrants_info (list or array) : a (M,10) shape list or array of all non empty quadrants.
        
        quadrant_vertices (array) : a (M,4,2) shape array of all vertices of quadrants.
                
    Returns:        
        quadrants_borders_info (array) : a (N,4,4) shape array of all border information of all infertile quadrants.
    """
    #Initialize quadrants borders info to keep neighhbors of quadrants
    quadrants_borders_info = np.zeros((len(quadrants_info),4,4), dtype=int)
    len_of_infertile_quadrants = len(indexes_of_infertile_quadrants)
    #Sort your different widths from smaller to bigger.
    sorted_widths = np.sort(np.unique(np.array(quadrants_info)[:,2]))
    #Keep length of different widths
    len_of_widths = len(sorted_widths)
    #Initialize lists to collect quadrants and their respective vertices by their width.
    quadrants_per_width_collection, quadrants_vertices_per_width_collection = [], []
    #For every different quadrant width in your quadtree.
    for w in sorted_widths:
        #Save the indexes collection of same width quadrants into your list.
        quadrants_per_width_collection.append(np.where(np.array(quadrants_info)[:,2]==w)[0])
        #Save the vertices collection of same width quadrants into your list.
        quadrants_vertices_per_width_collection.append(np.reshape(quadrant_vertices[quadrants_per_width_collection[-1]], (len(quadrants_per_width_collection[-1])*4,2)))
    #For index for quadrant without descendants
    for q_ind, ind in enumerate(indexes_of_infertile_quadrants):
        #Get info and vertices of quadrant
        q_info, q_vertices = quadrants_info[ind], quadrant_vertices[ind]
        #Initilize next width parameter
        next_width = 0
        #Find same size quadrants from quadtree.
        quadrant_width_index = np.where(sorted_widths==q_info[2])[0][0]
        while not np.all(quadrants_borders_info[ind]):
            #Check if you have exceeded the max width possible
            if len_of_widths <= quadrant_width_index+next_width:
                break
            #Gather the quadrants
            same_q_indexes = quadrants_per_width_collection[quadrant_width_index+next_width]
            #Gather all vertices of the same size quadrants
            same_q_vertices = quadrants_vertices_per_width_collection[quadrant_width_index+next_width]
            #For every vertex for your quadrant of interest
            for i, ver in enumerate(q_vertices):
                #Find common vertices
                touch = np.where((same_q_vertices==ver).all(axis=1))[0]
                #0 -> 1,2,3  ___  1 -> 2,3,0  ___  2 -> 3,0,1  ___  3 -> 0,1,2
                for vertex in touch:
                    #ssq_ind short for same size quadrant index with respect to same_q_indexes array
                    ssq_ind = int(vertex/4)
                    #Vertex point position can be one of (0,1,2,3)
                    vertex_position = int(((vertex/4)-ssq_ind)/0.25)
                    #If border had not been covered
                    if not quadrants_borders_info[ind][i][vertex_position]:
                        #Keep index of ssq according to its position in respect with your quadrant
                        quadrants_borders_info[ind][i][vertex_position] = same_q_indexes[ssq_ind]
            #Search for border at the next width size
            next_width += 1
            #Get info and vertices of the parent quadrant!
            q_info, q_vertices = quadrants_info[q_info[4]], quadrant_vertices[q_info[4]]
        #Print out update
        print(f'Borders of quadrant number {q_ind}/{len_of_infertile_quadrants} have been explored.', end='\r')
    #Return quadrants_borders_info
    return quadrants_borders_info

def compute_mec(quadrant, ordered_conductors, qp):
    """
    Get the MEC (maximum empty cube) for every conductor in the selected quadrant.
    
    Args:
        quadrant (class object) : quadrant object of the class Quadrants.
        
        ordered_conductors (list or array) : a (N,2) shape numpy array or list of all ordered quadrilateral of the dataset.
        
        qp (list or array) : a len (2) list or array of coordinates of query point which belongs to the selected quadrant.
                
    Returns:        
        mecs (list) : a len (M) list of all maximum empty cubes from query point to all conductors in the quadrant.
    """
    #Create list to keep MEC values
    mecs = []
    #Find the closest vertex of each conductor to the query point.
    shortest_vertices = np.argmin(np.linalg.norm(ordered_conductors[quadrant.unique_conductors_indexes]-qp, axis=2),axis=1)
    #Check if the point is in between the parallels of the conductors (not inside the conductor!!!)
    point_positions = check_if_point_between_parallels_of_set_of_conductors(ordered_conductors[quadrant.unique_conductors_indexes], qp)
    #For unique conductor index in 
    for cond_num, cond_ind in enumerate(quadrant.unique_conductors_indexes):
        conductor = ordered_conductors[cond_ind]
        #Check if the point is in between the parallels of the conductor (not inside the conductor!!!)
        point_position = point_positions[cond_num]
        #If point between parallels of quadrilateral
        if point_position[1] or point_position[2]:
            #If the point resides inside the region of the parallels of the vertical lines of the quadrpoint_positionral
            if point_position[0]:
                #If the point is over the top horizontal
                if point_position[1]>0:
                    #Compute the perpendicular distance of the point to the top horizontal
                    mec = abs(conductor[0][1]-qp[1])
                #Else the point is surely under the bottom horizontal
                else:
                    #Compute the perpendicular distance of the point to the bottom horizontal
                    mec = abs(conductor[3][1]-qp[1])
            #Else the point is guaranteed to be inbetween the parallels of the horizontal lines of the rectangular
            else:
                #If the point is to the right of the right vertical line of the rectangular
                if point_position[2]>0:
                    #Compute the perpendicular distance of the point to the right vertical
                    mec = abs(conductor[2][0]-qp[0])
                #Else the point is surely to left of the right vertical line
                else:
                    #Compute the perpendicular distance of the point to the left vertical
                    mec = abs(conductor[3][0]-qp[0])
        else:
            #Pick the closest vertex of the conductor to the query point.
            shortest_vertex = shortest_vertices[cond_num]
            #If the point is outside the rectangular region simply compute the l infinity from the closest vertex point.
            mec = np.max(abs(conductor[shortest_vertex]-qp))
        mecs.append(mec)
    return mecs

def qp_distance_from_borders(quad_vertices, qp):
    """
    Compute the distance of the query point from the four sides and the four corners of the quadrant that resides.
    
    Args:
        quad_vertices (array) : a (4,2) shape array of the four vertices of the selected quadrant.
        
        qp (list or array) : a len (2) list or array of coordinates of query point which belongs to the selected quadrant.
                
    Returns:        
        border_distances (list) : len(8) list of distances from the borders of quadrant to the qp, starting from left side to the left bottom corner, clockwise.
    """
    #Distance of point from sides of quadrant and distance of query point from corners of quadrant
    sides_distance = abs(quad_vertices-qp)
    corner_distance = np.max(sides_distance, axis=1)
    #Keep distances from border, from left side to left bottom corner, clockwise.
    border_distances = [sides_distance[0][0], corner_distance[0], sides_distance[0][1], corner_distance[1],
                        sides_distance[2][0], corner_distance[2], sides_distance[2][1], corner_distance[3]]
    return border_distances

def search_borders_conductors(qp, ordered_conductors, quadrant_vertices, quadrants, quadrants_borders, quad_ind,
                              mec_distance, closest_conductor, side_corner_to_quadrant_border, searched_quadrants=None):
    """
    Compute the distance of the query point from the four sides and the four corners of the quadrant that resides.
    
    Args:        
        qp (list or array) : a len (2) list or array of coordinates of query point which belongs to the selected quadrant.
        
        ordered_conductors (array) : a (M,2) shape numpy array of all conductors with ordered quadrilateral of the dataset.
        
        quadrant_vertices (array) : a (N,4,2) shape array of all vertices of quadrants.
        
        quadrants (list) : a len(N) list with all quadrant objects of class Quadrant, of the selected quadtree.
        
        quadrants_borders (array) : a (N,4,4) shape array of all border information of all infertile quadrants.
        
        quad_ind (int) : index of the quadrant that the query point resides.
                
        mec_distance (float) : distance of the query point from the closest conductor.
        
        closest_conductor (int) : index of the closest conductor to the query point.
        
        side_corner_to_quadrant_border (dictionary) : a dict that maps sides and corners of quadrant to indexes for quadrants_borders array.
        
        searched_quadrants (list or None) : default is None, optional a list of len(K) with searched indexes of quadrants.
                
    Returns:        
        
        closest_conductor (int) : index of the closest conductor to the query point.
        
        mec_distance (float) : distance of the query point from the closest conductor.
    """
    if not searched_quadrants:
        #Keep quadrant indexes that have been searched
        searched_quadrants = [quad_ind]
    #Keep also empty quadrants
    empty_quadrants = []
    #Find distances from borders (sides and corners, in total 8).
    border_distances = qp_distance_from_borders(quadrant_vertices[quad_ind], qp)
    #Sort distances of query point from borders
    sorted_border_distances_indexes = np.argsort(border_distances)
    #For distances of borders
    for border_ind in sorted_border_distances_indexes:
        #If you have smaller border distance than the minimum mec distance.
        if border_distances[border_ind]<mec_distance:
            #Border quadrant index, namely which side or corners mapped to quadrant border, which returns the correct index.
            neigbor_index = quadrants_borders[quad_ind][side_corner_to_quadrant_border[border_ind]]
            #If your quadrant has no neighbor in this side or corner continue
            if not neigbor_index:
                continue
            #If this quadrant has been searched continue
            if neigbor_index in searched_quadrants:
                continue
            #Keep neighbor index as searched
            searched_quadrants.append(neigbor_index)
            #Find neighbor quadrant
            neighbor_quadrant = quadrants[neigbor_index]
            #If neighbor quadrant has no conductors
            if not len(neighbor_quadrant.unique_conductors_indexes):
                #After the search of the neighbors conductors you have to check also the neighbors of the neighbors that did not contain any conductors.
                empty_quadrants.append(neigbor_index)
                continue
            #Compute mecs of conductors from neighbor quadrant
            border_mecs = compute_mec(neighbor_quadrant, ordered_conductors, qp)
            #Keep the index of closest conductor from the set of the unique conductors of the neighbor quadrant
            min_borders_mec = np.argmin(border_mecs)
            #Keep the index of the closest conductor from the whole set of conductors and the MEC distance from it.
            neighbors_closest_conductor, neighbors_mec_distance = neighbor_quadrant.unique_conductors_indexes[min_borders_mec], border_mecs[min_borders_mec]
            #If this distance is smaller than your already minimum distance
            if neighbors_mec_distance<mec_distance:
                #Replace information about closest conductor and MEC distance
                closest_conductor, mec_distance = neighbors_closest_conductor, neighbors_mec_distance
        else:
            break
    #If you have empty neighbors, check the neighbors of the neighbor to be quaranteed that you have found the MEC.
    for empty_quadrant_ind in empty_quadrants:
        closest_conductor, mec_distance = search_borders_conductors(qp, ordered_conductors, quadrant_vertices, quadrants, quadrants_borders, empty_quadrant_ind,
                                                                    mec_distance, closest_conductor, side_corner_to_quadrant_border, searched_quadrants)
    return closest_conductor, mec_distance

class LinePoints():
    """
    A class to represent the points of the line.

    ...

    Attributes
    ----------
    points : array
        A (5,1) numpy array of vertex points and ID of the quadrilateral.
    """
    def __init__(self, points):
        """
        Parameters
        ----------
        points : array
            A (5,1) numpy array of vertex points and ID of the quadrilateral.
        """
        self.x1, self.y1 = points[:2]
        self.x2, self.y2 = points[2:4]
        self.idx = points[-1]
        
    def __repr__(self):
        return '{}, {}: {}'.format(str((self.x1, self.y1)), str((self.x2, self.y2)), repr(self.idx))
    
    def __str__(self):
        return 'P1({:.2f}, {:.2f}), P2({:.2f}, {:.2f})'.format(self.x1, self.y1, self.x2, self.y2)
    
class Quadrant():
    """
    A class to represent the quadrants which partition the 2D space, the subdivided regions may be square or rectangular.

    ...

    Attributes
    ----------
    x0 : float
        The origin of the quadrilateral quadrant in x axis.
        
    y0 : float
        The origin of the quadrilateral quadrant in y axis.
        
    w : float
        The width of the quadrilateral quadrant (x axis dimension).
        
    h : float
       The height of the quadrilateral quadrant (y axis dimension).
        
    points : array
        A (5,1) numpy array of vertex points and ID of the quadrilateral.
        
    Methods
    -------
    get_width
        Returns the width of quadrilateral.
        
    get_height
        Returns the height of the quadrilateral.
        
    get_lines
        Returns the list of the LinePoints class's object.
    """
    def __init__(self, x0, y0, w, h, line_points):
        """
        Parameters
        ----------
        x0 : float
            The origin of the quadrilateral quadrant in x axis.

        y0 : float
            The origin of the quadrilateral quadrant in y axis.

        w : float
            The width of the quadrilateral quadrant (x axis dimension).

        h : float
           The height of the quadrilateral quadrant (y axis dimension).

        line_points : list
            List of objects of LinePoints class.
        """
        self.x0 = x0
        self.y0 = y0
        self.width = w
        self.height = h
        self.lines = line_points
        self.unique_conductors_indexes = np.unique([line.idx for line in self.lines]).astype(int)
        self.children = []

    def get_width(self):
        '''
        Returns the width of quadrilateral.
        '''
        return self.width
    
    def get_height(self):
        '''
        Returns the height of the quadrilateral.
        '''
        return self.height
    
    def get_lines(self):
        '''
        Returns the list of the LinePoints class's object.
        '''
        return self.lines
    
    def get_info(self):
        '''
        Returns a list with unique info of the quadrant
        '''
        return [self.x0, self.y0, self.width, self.height]
    
    def __str__(self):
        return print(f'x0 = {self.x0}, y0 = {self.y0}, width = {self.width}, height = {self.height}, number_of_lines_included = {len(self.lines)}')
    
class QTree():
    """
    The class that initialize quadtree parameters and executes the subdivision of the quadrants.
    
    ...

    Attributes
    ----------
    points : array
        A (5,1) numpy array of vertex points and ID of the quadrilateral.
    
    k : int
        Threshold for the number of lines that can be found inside a quadrant.
        
    root_node: list or tuple
        List or tuple with the information of the starting quadrilateral, needs x0, y0, width and height.
        For more see the description of Quadrant class.
        
    Methods
    -------
    get_lines
        Returns the list of the LinePoints class's object.
        
    subdivide
        Builds the quadtree structure by subdividing overcrowded (len(lines)>k) parental quadrants until the set threshold is satisfied.
    
    graph
        Function to plot the quadtree structure and the points of the line.
    """
    def __init__(self, lines, k, root_node):
        """
        Parameters
        ----------
        points : array
            A (5,1) numpy array of vertex points and ID of the quadrilateral.

        k : int
            Threshold for the number of lines that can be found inside a quadrant.

        root_node: list or tuple
            List or tuple with the information of the starting quadrilateral, needs x0, y0, width and height.
            For more info see the description of Quadrant class.
        """
        self.threshold = k
        self.lines = [LinePoints(line) for line in lines]
        self.root = Quadrant(*root_node, self.lines) #Quadrant(0, 0, 50, 50, self.lines)
    
    def get_lines(self):
        '''
        get_lines
            Returns the list of the LinePoints class's object.
        '''
        return self.lines
    
    def subdivide(self):
        '''
        subdivide
            Builds the quadtree structure by subdividing overcrowded (len(lines)>k) parental quadrants until the set threshold is satisfied.
        '''
        recursive_subdivision(self.root, self.threshold)
    
    def graph(self):
        '''
        graph
            Function to plot the quadtree structure and the points of the line.
        '''
        fig = plt.figure(figsize=(12, 8), dpi=200)
        plt.title("Quadtree")
        c = find_children(self.root)
        print("Number of segments: %d" %len(c))
        areas = set()
        for el in c:
            areas.add(el.width*el.height)
        print("Minimum segment area: %.3f units" %min(areas))
        for n in c:
            plt.gcf().gca().add_patch(patches.Rectangle((n.x0, n.y0), n.width, n.height, fill=False))
        x1 = [point.x1 for point in self.lines]
        y1 = [point.y1 for point in self.lines]
        plt.plot(x1, y1, 'ro', markersize=0.5) # plots the points as red dots
        plt.show()
        plt.close()
        return
    
#Maps sides and corners of quadrant to indexes for quadrants_borders array
side_corner_to_quadrant_border = {0:(0,1), 1:(0,2), 2:(0,3), 3:(1,3), 4:(1,0), 5:(2,0), 6:(2,1), 7:(3,1)}