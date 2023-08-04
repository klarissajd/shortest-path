import heapq
import math

__author__ = "Klarissa Jutivannadevi"
__student_id__ = "32266014"


# PROBLEM 1 - Gotta Go Fast


class RoadGraph:
    """
    A class created for Question 1. The RoadGraph class is a class that contains
    an adjacency list for a graph representation that contains roads for different
    vertices and also the cafes and also the methods that are involved in order to
    produce the solution for routing function. Adjacency list is used instead of adjacency
    matrix due to the overall complexity that they result in (which is most of the time more
    efficient than doing with adjacency matrix)
    :Attributes:
        vertices: The total vertices in a graph
        cafe_and_time: the waiting time of each cafe (which is stored as a
        list where each index is the cafe at certain location and the value
        is the waiting time)
        adjacency_list: a graph of road stored as a list of list where
        each index is the starting and the first index of the inner list
        is the ending and the second index is the weight of the edge
        reverse_adjacency_list: Exactly the same as adjacency_list, with
        a difference of positioning where it is assumed that the direction
        is reversed (starting point becomes the ending point and vice versa)
    """

    def __init__(self, roads, cafe):
        """
        This is a constructor for the RoadGraph class which creates a directed
        graph from one vertex (road) to another vertex by the use of adjacency
        list. This is also used to create a list for cafe waiting time, where
        each time is put based on the cafe location.
        :Input:
            :param roads: A tuple where index 0 and 1 is the vertex where it is from
            and where it is going. Index 2 is to show the distance. This gives a hint
            that the graph directed weighted
            :param cafe: Gives a list of tuple where the index 0 is the cafe on the
            certain road (vertex) and the waiting time to get the coffee.
        :Time complexity: O(V+E) where either O(V) is the highest when empty adjacency
        list is created and the total outer list is based on the total existing location
        or O(E) where roads (edge) is iterated in the list in order to put the edge to
        the adjacency list. Since there is no guarantee that E > V or vice versa, complexity
        is O(V+E)
        :Aux space complexity: O(V+E) where the adjacency list is created. This adjacency list
        can contain an inner list of edges based on the outer list index indicating the vertex.
        """

        self.vertices = self.total_vertex(roads) + 1  # number of vertices
        self.cafe_and_time = self.cafes_waiting_time(cafe, self.vertices)  # the list of time arranged based on vertex

        # creating an adjacency list
        # TC: O(V) where V is the time taken for the looping of no. of vertices to create
        # a new list within to store edges
        # SC: O(V+E) where V is the number of vertices in the graph since each element will
        # hold the ending vertex together with the weight once an edge connected to it is
        # created and E is the edges that will be contained in the inner list
        self.adjacency_list = [[] for _ in range(self.vertices)]
        # creating a reverse adjacency list (for dijkstra reverse implement from end to certain point)
        # I have noted that this question should use directed graph
        # Understood that but for implementation purpose this will be needed (and will be explained later)
        self.reverse_adjacency_list = [[] for _ in range(self.vertices)]

        # TC: O(E) where E is the edges to be in adjacency list where it is iterated
        # in every index in the roads list. The complexity of add_edge is constant.
        # SC: O(1) since no new list is created
        for road in roads:
            # adding an edge to the graph
            self.add_edge(self.adjacency_list, road[0], road[1], road[2])
            # adding an opposite edge to a similar graph
            self.add_edge(self.reverse_adjacency_list, road[1], road[0], road[2])

    def total_vertex(self, road):
        """
        This method is used to find the highest vertex in the road.
        The biggest value can be used since we can assume that the
        vertex is from 0 to |V|-1 which means that the highest input inside
        either index 0 or index 1 from the list of list road is the maximum
        vertex.
        :Input:
            :param road: The input road with the starting and ending vertex and weight
            of the edge (but only index 0 and index 1 of road[i] is considered since
            only the vertex is needed
        :return: The largest vertex found
        :Time complexity: O(E) where E is the edges in the road iterated to find
        the largest vertex
        :Aux space complexity: O(1) as only a variable is used to store value
        (in-place)
        """
        total = 0       # variable storing an integer has SC: O(1)
        # TC: O(E) where E is the edges (since road contains the edges)
        for i in road:
            # find the biggest vertex
            if total < i[0]:
                total = i[0]
            elif total < i[1]:
                total = i[1]
        return total

    def cafes_waiting_time(self, cafe, vertex):
        """
        The method creates an empty array based on the vertex which indicates the location.
        We know that in certain location there is a cafe based on the parameter passed.
        So, each waiting time will be allocated based on the vertex (to mark that there
        is a cafe in that location)
        :Input:
            :param cafe: The cafe array consisting of the tuple (vertex, waiting time). This is
            the exact same input as the ones passed on to __init__ which has a waiting time
            for certain location
            :param vertex: How many vertex are in the graph
        :return: An array consisting each waiting time in the index (based on vertex)
        :Time complexity: O(V) where V is the maximum time taken to iterate the list of cafe
        (since each location can either have a SINGLE cafe or not at all) which is passed as
        a tuple together with the waiting time
        :Aux space complexity: O(V) where V is the number of vertices (which will be used to mark
        an existence of a cafe (based on its waiting time) for every location)
        """
        # SC: O(V) where V is the length of the vertex created to store waiting time
        # TC: O(V) where V is iterated to create a list with initial value infinity
        # in each index reaching iteration V
        cafe_time = [math.inf for _ in range(vertex)]
        # TC: O(V) where V is the length of the cafe (can only be maximum of V since a location
        # can only have 1 cafe at most (since it is unique))
        for i in range(len(cafe)):
            # handling the condition in which cafe does not exist in any possible location
            if cafe[i][0] > len(cafe_time) - 1:
                return cafe_time
            # change infinity to real waiting time based on the param cafe indicating that cafe exists there
            else:
                cafe_time[cafe[i][0]] = cafe[i][1]
        return cafe_time

    def add_edge(self, adj_list, u, v, w):
        """
        Adding an edge to the adjacency list in order to update the edge
        of the graph together with its distance. u, v, w is just the elements
        separated by index from the original tuple passed on by road param indicating
        the edges
        :Input:
            :param adj_list: the adjacency list of a graph to be modified (have a new edge added)
            :param u: the starting vertex
            :param v: the ending vertex
            :param w: the weight of the edge (distance between u and v)
        :return: does not return anything but have the adj_list updated each time with this method
        being called
        :Time complexity: O(1) by appending the edge to the end of the list inside
        the list (based on the starting vertex)
        :Aux space complexity: O(1) since no new space is created
        """
        adj_list[u].append([v, w])

    def dijkstra(self, start, lst):
        """
        source: dijkstra pseudocode in notes.pdf with almost no difference except for
        the priority queue with only an initial vertex instead of putting all vertices.
        Dijkstra's Algorithm is the algorithm that is used to find the shortest path
        of the start to the rest of the vertex and will work with cycle as long as there
        are no negative cycles. Since the road are not dealing with negative distance, Dijkstra's
        algorithm can be used. This algorithm works by using Priority Queue where each priority
        are based with the nearest distance. And distance to the entire vertices are updated once
        it meets an edge leading to a shorter distance which will also update the predecessor.
        :Input:
            :param start: The starting vertex (to start finding the path to the rest
            of the vertices)
            :param lst: The adjacency list which is used to do the dijkstra algorithm
        :return:
            dist: A list containing the shortest path from the starting vertex to the rest of the
            vertex arranged by the vertices (which is by the index)
            pred: A list containing the predecessor of every vertex leading to the shortest path
            (also each vertex is arranged by index)
        :Time complexity: The time complexity of the algorithm would be
        O(|E|log|V|) where visiting each edges took a complexity of O(E) and O(log|V|) comes from
        pushing the new queue to the Priority Queue and reordered by the priority
        :Aux space complexity: The auxiliary space complexity would be O(|V|) where O(|V|) is
        the dist and the pred each created to store the distance and predecessor of each vertex
        respectively.
        """
        # assigning dist to be inf for all vertex and starting dist to be 0
        dist = [math.inf] * self.vertices
        dist[start] = 0
        # creating the predecessor (-1 to annotate no predecessor)
        pred = [-1] * self.vertices

        # initialize the start location which is put to the queue
        priority_queue = [[start, 0]]

        # keep on looping while priority queue is not empty
        while not len(priority_queue) == 0:
            n_vertex, n_dist = heapq.heappop(priority_queue)

            # loop through all the adjacency list that has edge to the current u
            # adjacency_list[u_vertex] is from the list inside the adjacency list (so it is [end, dist], ...)
            # a is ending vertex
            # b is the distance to the ending vertex
            # TC: O(E) where E is the edges iterated in certain vertex to find the shortest path
            for a, b in lst[n_vertex]:
                # compare whether the current distance on dist is greater than the new computation from the edge
                if dist[a] > n_dist + b:
                    dist[a] = n_dist + b
                    pred[a] = n_vertex
                    # TC: O(log|V|) where heap-up operation occur
                    heapq.heappush(priority_queue, [a, dist[a]])
        return dist, pred

    # finding the optimal route
    def routing(self, start, end):
        """
        The routing method is the method that find the closest time from start location to end location and at
        the same time allow a coffee to be ordered at a cafe and still gives the shortest time. This can be done
        by executing Dijkstra's Algorithm twice where the cafe will be the midpoint and the time from start to cafe
        and cafe to end will be added together in order to produce the shortest time possible.
        :Input:
            :param start: The starting point where a path will start from
            :param end: The ending point where that point should be reached from the starting point
        :return: A list of path to the locations where they need to visit in order to grab a coffee from starting point
        to ending point in the shortest time
        :Time complexity: O(Elog|V|) where this comes from Dijkstra's Algorithm being executed twice for 2 routes
        (to be precise O(2Elog|V|)). The other complexities taken from other method only reaches to a linear complexity,
        hence is not considered.
        :Aux space complexity: O(V+E) where V is the total locations that existed in the graph and E is the edges
        connected in the graph since E can be lesser than V (in case if everything is connected, E can at least be
        V-1 which is essentially smaller than V). The list created in this method is actually only the path, which has
        a space complexity of O(V). But, routing() also takes other method to compute its final returned value and some
        have a complexity of O(V+E). Since we cannot assume that E will always be greater than V (e.g. if graph is
        disconnected), it is best to say that the overall space complexity of this is O(V+E)
        """
        # handle the condition where the starting point or ending point is not within the available location
        if start > self.vertices - 1 or end > self.vertices - 1:
            return None

        # TC: O(ElogV) dijkstra algorithm is used
        # this is used to trace starting point to all the other points
        start_dist, start_pred = self.dijkstra(start, self.adjacency_list)
        # this is used for ending pint to all other points (hence the use of graph containing reverse edges
        # is needed)
        end_dist, end_pred = self.dijkstra(end, self.reverse_adjacency_list)

        # total time to calculate all time (starting location to cafe + cafe to ending location + cafe waiting time)
        # SC: O(V) where each element will hold the total time of each location
        total_time = [0] * self.vertices

        # counting the total time
        # the overall path that does not visit any cafe is considered as irrelevant hence becomes infinity
        # the index of total_time represent the cafe
        # TC: O(V) where V is the locations (where the cafe are within the existing location)
        for i in range(self.vertices):
            total_time[i] = self.cafe_and_time[i] + start_dist[i] + end_dist[i]

        # finding the smallest time based on everything computed
        current_low = math.inf
        # the location of the cafe that contains the smallest overall time
        lowest_cafe = -1

        # TC: O(V) where V is the length of the vertex (cafe is within the location)
        for i in range(self.vertices):
            if total_time[i] < current_low:
                lowest_cafe = i
                current_low = total_time[i]
        # if the smallest time is still infinity, it means that there is no cafe. Hence, no existing path
        if current_low == math.inf:
            return None

        # get the preceding vertex starting from cafe until it reaches the starting position
        preceding = start_pred[lowest_cafe]
        # get the succeeding vertex starting from cafe up to the ending position (based on reverse-edge graph)
        succeeding = end_pred[lowest_cafe]

        # empty array to store result of the path taken
        travelled = []

        # TC: O(V) where V is the preceding vertex (since each vertex are visit once, worst case is when all vertex are
        # the predecessor of another vertex)
        while preceding != -1:
            travelled.append(preceding)
            preceding = start_pred[preceding]

        # reverse the order first to fix the array form
        # TC: O(V) where V//2 is the maximum possible length and that only happens when in case the starting point
        # goes through all the point before reaching the cafe
        # SC: O(1) since only swapping is done to reverse the order and no new list is created to contain this
        for i in range(len(travelled) // 2):
            travelled[i], travelled[(len(travelled) - 1) - i] = travelled[(len(travelled) - 1) - i], travelled[i]
        # appending to last only require a O(1) TC
        travelled.append(lowest_cafe)

        # TC: O(V) where V is the locations. This has the same logic as the preceding.
        # Hence, in Worst_Case: succeeding = O(V) + preceding = O(1) where the succeeding contains all vertex
        # and preceding does not contain a vertex
        # In succeeding best case, preceding will have the worst case. This follows the same logic as the worst case
        # of succeeding but reversed idea
        while succeeding != -1:
            travelled.append(succeeding)
            succeeding = end_pred[succeeding]

        # final checking to make sure that if no path are traced, return path is None
        if len(travelled) == 0:
            travelled = None

        return travelled


# --------------------------------------------
# QUESTION 2 - Maximum Score


def add_edge(adj_list, u, v, w):
    """
    p.s. copied from the comment above since it is the exact same function
    Adding an edge to the adjacency list in order to update the edge
    of the graph together with its distance
    :Input:
        :param adj_list: the adjacency list of a graph to be modified (have a new edge added)
        :param u: the starting vertex
        :param v: the ending vertex
        :param w: the weight of the edge (distance between u and v)
    :return: does not return anything but every time this function is called, adj_list will have
    an addition of edge to its list
    :Time complexity: O(1) by appending the edge to the end of the list inside
    the list (based on the starting vertex)
    :Aux space complexity: O(1) since no new space is created
    """
    adj_list[u].append([v, w])


def total_vertex(downhillRoute):
    """
    This method is used to find the highest vertex in the downhillRoute.
    The biggest value can be used since we can assume that the
    vertex is from 0 to |V|-1 which means that the highest input inside
    either index 0 or index 1 from the list of list downhillRoute is the maximum
    vertex.
    :Input:
        :param downhillRoute: The input road with the starting and ending vertex and weight
        of the edge (but only index 0 and index 1 of downhillRoute[i] is considered since
        only the vertex is needed)
    :return: The largest vertex found
    :Time complexity: O(E) where E is the edges in the downhillRoute iterated to find
    the largest vertex
    :Aux space complexity: O(1) as only a variable is used to store value
    (in-place)
    """
    total = 0
    # TC: O(D) where D is the edges and is iterated in order to find the highest existing vertex
    for i in downhillRoute:
        if total < i[0]:
            total = i[0]
        elif total < i[1]:
            total = i[1]
    return total


def topological_sort(adj_list, u, visited, stack, vertices):
    """
    source: https://www.geeksforgeeks.org/shortest-path-for-directed-acyclic-graphs/
    with some added parameter and changed logic in order to find the maximum instead
    of the minimum distance and accept required parameter since it does not use a
    class.
    The topological sort method is used since it allow any directed graph as long as
    it does not contain a cycle. In this algorithm, a vertex with no incoming edge will be ordered
    first and then this will keep on traversing until the last vertex following this condition where
    there is no more incoming edges. This sorting is used instead of Bellman-Ford Algorithm
    since Bellman-Ford needs to iterate repeatedly to ensure the minimum (in this case maximum)
    weight is true which ends up in taking a O(DP) complexity. Since we can assume that cycle
    does not exist, we did not need to check for cycle, hence topological sort will be more efficient.
    :Input:
        :param adj_list: Takes the adjacency list based on the starting vertex to the ending vertex
        and its weight
        :param u: The current vertex that will have the edges adjacent to it checked
        :param visited: The vertices that had its edges checked
        :param stack: The order in which the "children" and there is no other incoming edge
        from that vertex
        :param vertices: The total vertices that exist in the graph
    :return: Technically does not return anything, but has its stack updated each time which
    is needed later.
    :Time complexity: O(D) which will be explained later. Meanwhile, can be thought of a bigger
    complexity of O(D+P) where D is the edges being iterated once based on the vertex and P
    is the vertex that goes recursively.
    :Aux space complexity: O(1) since stack is only appended here and not created
    """
    visited[u] = True
    if u < vertices:
        # TC: O(D) where the edge of the adjacency list is visited one by one until it reaches the
        # smallest where there is no outgoing edges. Similar to DFS where it searches to the last vertex
        # that is unreachable to other vertex anymore and then trace back.
        for v, w in adj_list[u]:
            if visited[v] is False:
                topological_sort(adj_list, v, visited, stack, vertices)
    stack.append(u)


def shortest_path(start, vertices, adj_list):
    """
    source: https://www.geeksforgeeks.org/shortest-path-for-directed-acyclic-graphs/ where this
    is a continuation of the method above.
    This is a Direct Acyclic Graph (DAG) algorithm which takes the stack ordering that has been done
    in topological order and then start calculating the shortest distance once all the vertex has been
    iterated. It updates the distance (score) and the predecessor based on the newest maximum score gathered
    :Input:
        :param start: The starting vertex where the path will start
        :param vertices: The total number of vertices in the graph
        :param adj_list: The adjacency list that holds the graph
    :return: The distance from start to all other vertex (where the index means the reaching vertex)
    and the predecessor which holds the preceding vertex of that certain vertex (known by the index)
    :Time complexity: O(D). This is actually an O(D+P) complexity where P is the intersection point. But in this
    case, all the points should be connected and not a cycle (since by the assignment pdf we are allowed to assume
    that each intersection point (P) has at least a downhill segment (D) that start or finish at that intersection
    point. This means that the minimum edge that is permitted is only P-1. Even if P is possibly larger than D,
    the largest it can be is only +1 of D. Hence, we can just assume complexity to be O(D+P) = O(D+(D+1)) which gives
    O(2D+1) leading to a general complexity of O(D).
    :Space complexity: O(P) where P is the total vertex. This happens since stack will only store each vertex
    once (in which happens when no outgoing edge is detected). This means that eventually at most all vertex
    will only be included once. The array used to check the visited vertex is also index based. Hence, an exact
    length of P will always be generated for the visited list. Same goes to dist and scores.
    """
    # SC: O(P) where P is the number of vertices
    visited = [False] * vertices
    # SC: O(P) where P is the number of vertices put in order (vertex with no outgoing edge is added here)
    stack = []

    # TC: O(D) where D is the edges in the graph
    # although this seems like a quadratic time complexity, it is in fact linear. This is because in this
    # question, all the vertices is connected which means that only visited[0] is computed and gives a complexity
    # within of O(D) where stack is updated. But in the next iteration, visited[i] will be True. This means that the
    # condition is skipped. This means that the complexity will roughly be O(P) + O(1) * O(D) which equivalents to
    # O(P+D) and since D is mostly bigger than P, it can be thought as O(D)
    for i in range(vertices):
        if visited[i] is False:
            topological_sort(adj_list, start, visited, stack, vertices)

    # SC: O(P) where P is the total vertices
    dist = [-math.inf] * vertices
    dist[start] = 0
    # SC: O(P) where P is the total vertices
    # uses -1 as the default since smallest vertex indicator is 0, hence we know that -1 is not a legit vertex
    pred = [-1] * vertices

    # TC: O(D+P) where the vertex only goes through once and the edges adjacent to that vertex follows
    # O(P) comes from the stack and O(D) comes from the for loop within
    # reminder this is not O(DP) since the edges visit is ONLY the edges adjacent to that current vertex
    while stack:
        i = stack.pop()
        for v, w in adj_list[i]:
            if dist[v] < dist[i] + w:
                dist[v] = dist[i] + w
                pred[v] = i

    return dist, pred


def optimalRoute(downhillScores, start, finish):
    """
    The optimalRoute is the main method that will be called to generate the result of the maximum score
    obtained. This method simply compiles the path that was taken from the preceding based on the above
    method which already had its highest score calculated.
    :Input:
        :param downhillScores: The list given that contains the list of tuple containing starting point, ending point
        and the scores obtain from that path
        :param start: The starting point of the path
        :param finish: The ending point of the path to be reached
    :return: A list consisting of the path that is taken in order to reach the finishing path from the starting path
    that earns the most scores
    :Time complexity: O(D) where the greatest complexity comes from getting the shortest path (based on the comments
    on the shortest_path() method which has the time complexity simplified). The for loop to iterate through all
    the edges to be appended to the adjacency list also takes O(D) complexity, which is the same complexity but has
    a SLIGHTLY smaller complexity when is compared in detailed to the shortest_path().
    :Aux space complexity: O(D) where a list of list is created to store the edges. Similar to the time complexity,
    this can be considered as O(D+P) but since D can only have the least value of -1 less than P, it can be assumed as
    O(D) instead. And in all case except for one (which D is P-1), D is >= P. Hence the reason why.
    p.s. some comments below might state SC as O(P) but this can be thought of the worst space complexity assuming it
    is equivalent to O(D+1) but regardless can still be considered as O(D) since +1 is constant meaning that
    it is irrelevant. Writing it as O(P) is just more reasonable in some parts and so will be written as that way.
    """
    vertices = total_vertex(downhillScores) + 1
    # TC: O(P) where P is the total vertices (iterated to create empty list each time within a bigger list)
    # SC: O(D+P) where P is the intersection point and D is te edges to be appended to create an adjacency list
    # hence is just equivalent to O(D) is general BigO space complexity
    graph = [[] for _ in range(vertices)]

    # TC: O(D) where D is the edges that is appended to the adjacency list
    for i in downhillScores:
        # appending only takes O(1)
        add_edge(graph, i[0], i[1], i[2])

    # TC: O(D) based on the shortest_path() method. Same SC applies
    score, predecessor = shortest_path(start, vertices, graph)

    # check for existing path (if no existing path, weight will be -infinity)
    if score[finish] == -math.inf:
        return None

    # SC: O(P) where P is the greatest possible length and happens if all path needs to be passed
    # to reach the end from start
    path = []

    # TC: O(P) where P is the total vertex and the maximum iteration before predecessor of -1 is reached
    # (which is the predecessor of the starting point)
    while finish != -1:
        path.append(finish)
        finish = predecessor[finish]

    # TC: O(P) where P is the total vertex that has been compiled in the path
    # SC: O(1) since no new list is introduced, the previously created list is just swapped in order to reverse
    # the order (since the end is taken first and traced back using its predecessor to reach the start)
    for i in range(len(path) // 2):
        path[i], path[len(path) - 1 - i] = path[len(path) - 1 - i], path[i]

    return path


# if __name__ == '__main__':
#     roads_basic = [(0, 1, 4), (1, 2, 2), (2, 3, 3), (3, 4, 1), (1, 5, 2),
#                    (5, 6, 5), (6, 3, 2), (6, 4, 3), (1, 7, 4), (7, 8, 2),
#                    (8, 7, 2), (7, 3, 2), (8, 0, 11), (4, 3, 1), (4, 8, 10)]
#     cafes_basic = [(5, 10), (6, 1), (7, 5), (0, 3), (8, 4)]
#
#     rg = RoadGraph(roads_basic, cafes_basic)
#     print(rg.routing(3, 4))
