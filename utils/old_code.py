def check_if_point_between_parallels_of_quadrilateral(quad,q):
    """
    Check if a point resides inside the region of the two parallel sets of sides of a quadrilateral.

    Args:
        quad (array): a (4,2) numpy array of vertex points.
        q (array or list): (2) or (1,2) numpy array or list of a query point's coordinates in R^2 space.
    Returns:
        False if the point is outside of the parallel sets region or
        the position of the point in respect with the four sides of the rectangular in the case that is inside the parallels region.
    """
    #Compute the position of the query towards vertical lines of quadrilateral
    left, right = position_of_point_towards_line(quad[3],quad[0],q), position_of_point_towards_line(quad[2],quad[1],q)
    #Compute the position of the query towards horizontal lines of quadrilateral
    top, bottom = position_of_point_towards_line(quad[1],quad[0],q), position_of_point_towards_line(quad[2],quad[3],q)
    #If the point is between the paralles of the vertical line segments
    if left*right<0:
        #Return 1 and top and bottom orientations
        return (1,top,bottom)
    elif top*bottom<0:
        #Return 0 and left and right orientations
        return (0,left,right)
    return False

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
    #For unique conductor index in
    for cond_num, cond_ind in enumerate(quadrant.unique_conductors_indexes):
        conductor = ordered_conductors[cond_ind]
        #Check if the point is in between the parallels of the conductor (not inside the conductor!!!)
        point_position = check_if_point_between_parallels_of_quadrilateral(conductor,qp)
        #If point between parallels of quadrilateral
        if point_position:
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

### Approach without for loop for mec computation, slower and wrong!!! ###

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
    return point_positions

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
    mecs = np.zeros(len(quadrant.unique_conductors_indexes))
    #Find the closest vertex of each conductor to the query point.
    shortest_vertices = np.argmin(np.linalg.norm(ordered_conductors[quadrant.unique_conductors_indexes]-qp, axis=2),axis=1)
    #Check if the point is in between the parallels of the conductors (not inside the conductor!!!)
    point_positions = check_if_point_between_parallels_of_set_of_conductors(ordered_conductors[quadrant.unique_conductors_indexes], qp)
    #Point is between the parallels of the vertical line segments for conductors.
    between_verticals = np.where((point_positions[0]*point_positions[1])<0)[0]
    #Point is between the parallels of the horizontal line segments for conductors.
    between_horizontal = np.where((point_positions[2]*point_positions[3])<0)[0]
    #Over the top horizontal, while in between vertical sides.
    over_top_horizontal = np.intersect1d(np.where(point_positions[2,:]>0), between_verticals)
    #To the right of the right vertical side, while in between horizontal sides.
    rightward_to_right_vertical = np.intersect1d(np.where(point_positions[1,:]>0), between_horizontal)
    #compute mecs
    quad_conductors = ordered_conductors[quadrant.unique_conductors_indexes]
    #
    mecs = np.max(abs(quad_conductors[:,shortest_vertices][0]-qp), axis=1)
    #
    mecs[between_verticals] = abs(quad_conductors[between_verticals,0][:,1]-qp[1])
    #
    mecs[over_top_horizontal] = abs(quad_conductors[over_top_horizontal,3][:,1]-qp[1])
    #
    mecs[between_horizontal] = abs(quad_conductors[between_horizontal,2][:,0]-qp[0])
    #
    mecs[rightward_to_right_vertical] = abs(quad_conductors[rightward_to_right_vertical,3][:,0]-qp[0])
    #
    #Where mecs non zero
    #for_l_infinity = np.where(mecs==0)
    #Pick the closest vertex of the conductor to the query point.
    #shortest_vertex = shortest_vertices[for_l_infinity]
    #Compute l infinity
    #mecs[for_l_infinity] = np.max(abs(ordered_conductors[quadrant.unique_conductors_indexes][for_l_infinity,shortest_vertex][0]-qp), axis=1)
    return mecs

### Approach without for loop for mec computation, slower and wrong!!! ###


### Old random query point creation functions ###

def check_if_point_inside_quadrilateral(quad,q):
    """
    Check if a point resides inside the region of the two parallel sets of sides of a rectangular.

    Args:
        quad (array): a (4,2) numpy array of vertex points.
        q (array or list): (2) or (1,2) numpy array or list of a query point's coordinates in R^2 space.
    Returns:
        True if the point is inside the quadrilateral
        False if the point is outside of the quadrilateral
    """
    left, right = position_of_point_towards_line(quad[3],quad[0],q), position_of_point_towards_line(quad[2],quad[1],q)
    top, bottom = position_of_point_towards_line(quad[1],quad[0],q), position_of_point_towards_line(quad[2],quad[3],q)
    if (left*right)<0 and (top*bottom)<0:
        return True
    return False

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
        #Keep the quadrilaterals that are inside your quadrant that your point was found.
        quads_ind_of_quadrant = np.unique([line.idx for line in quadrants[quadrant_ind].lines]).astype(int)
        #For every quadrilateral
        for quad_ind in quads_ind_of_quadrant:
            #Check if the point is inside it
            if check_if_point_inside_quadrilateral(ordered_conductors[quad_ind],random_point):
                #If indeed its true, change inside variable to True and break
                inside = True
                break
        #If inside is False
        if not inside:
            #Append your random point to your random_points list
            random_points.append(random_point)
            print(f'Have been created {len(random_points)}/{n_points} random query points.', end='\r')
    #When out of the while loop return the list with random points
    return random_points

### Plot Times ###

## LOG ##
plt.figure(figsize=(10,8), dpi=200)
plt.plot(k,(k*len(ordered_conductors))/15e+4, '^-',color='tab:red', lw=2, label='Complexity m*n')
plt.plot(k[:len(save_time_brut)],save_time_brut,'o-',color='tab:blue', lw=2, label='Brute Force')
plt.plot(k,(k*k)/15e+4,'^-', lw=2, color='tab:orange', label='Complexity n*n')
plt.plot(k,save_time, 'o-', lw=2,color='tab:cyan', label='Quadtree')
plt.plot(k,(k*np.log(k))/15e+4,'^-', lw=2,color='tab:green', label='Complexity n*log(n)')
plt.legend(loc='upper left')
plt.yscale('log')
plt.xlabel('Number of Query Points', fontsize=14)
plt.ylabel('log(Seconds)', fontsize=14)
plt.ylim((10e-5,10e+4))
plt.title('Query Points vs log(Seconds)')
plt.savefig(f'./Times_with_complexity.png')
plt.show()
plt.close()
## LOG ##

plt.figure(figsize=(10,8), dpi=200)
plt.plot(k[:len(save_time_brut)],np.array(save_time_brut)/60,'o-',color='tab:blue', lw=2, label='Brute Force')
#plt.plot(k_100k,save_time_100k, 'o-', color='tab:cyan', lw=2, label='Quadtree')
plt.legend(loc='upper left')
plt.xlabel('Number of Query Points', fontsize=14)
plt.ylabel('Minutes', fontsize=14)
plt.xlim((-50,1050))
plt.ylim((-5,105))
plt.title('Query Points vs Minutes')
plt.savefig(f'./Brute_force_times.png')
plt.show()
plt.close()
