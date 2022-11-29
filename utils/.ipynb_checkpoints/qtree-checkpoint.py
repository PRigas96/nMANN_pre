import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

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