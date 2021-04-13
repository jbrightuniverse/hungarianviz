"""
Hungarian Algorithm No. 5 by James Yuming Yu
Vancouver School of Economics, UBC
8 March 2021
Based on http://www.cse.ust.hk/~golin/COMP572/Notes/Matching.pdf and https://montoya.econ.ubc.ca/Econ514/hungarian.pdf

Modified for visual progression support
12 April 2021
"""
import numpy as np
from PIL import Image, ImageDraw, ImageFont

class Node:
    """A simple node for an alternating tree."""
    
    def __init__(self, val, parent = None):
        self.val = val
        self.parent = parent


INITIAL_STAGE = 0
UPDATE_STAGE = 1
NEIGHBOUR_STAGE = 2
MATCHING_STAGE = 3
FLIPPING_STAGE = 4
RESET_STAGE = 5
EXPANSION_STAGE = 6
EXIT_STAGE = 7


def process_frames(result, matrix):
    """Processes intermediate algorithm frames to construct images."""
    output_images = []

    font = ImageFont.truetype("OpenSansEmoji.ttf", 40)
    #font2 = ImageFont.truetype("OpenSansEmoji.ttf", 30)
    font3 = ImageFont.truetype("OpenSansEmoji.ttf", 20)

    for frame in result:
        # Prep render engine and frame size
        img = Image.new("RGBA", (500, 500), (255, 255, 255, 255))
        d = ImageDraw.Draw(img)
        singular = d.textsize("0" * max(len(str(np.amax(matrix))), 3), font = font)
        singular_width = singular[0] + 2
        singular_height = singular[1] + 2
        matrix_width = len(frame[0]) * singular_width
        matrix_height = len(frame[0]) * singular_height

        wide = d.textsize("0" * max(len(str(max(frame[0]))), 3), font = font)[0]
        img = img.resize((max(wide * 2 + matrix_width, 422), max(matrix_height + singular_height * 2 + 100, 346)))
        d = ImageDraw.Draw(img)

        d.text((2, img.height - 40), "(Red Dot: free row/col; Blue Dot: path entry)", fill = (0, 0, 0, 255), font = font3)
        if frame[4] == INITIAL_STAGE:
            d.text((2, img.height - 100), "Initial Stage: pick a free row", fill = (0, 0, 0, 255), font = font3)

        elif frame[4] == UPDATE_STAGE:
            d.text((2, img.height - 100), f"No Free Cells: Potential Update α = {frame[5]}", fill = (0, 0, 0, 255), font = font3)
            d.text((2, img.height - 70), "Purple: Added Cell", fill = (0, 0, 0, 255), font = font3)

        elif frame[4] == NEIGHBOUR_STAGE:
            d.text((2, img.height - 100), "Search For Equality Graph Adjacent to Tree", fill = (0, 0, 0, 255), font = font3)
            d.text((2, img.height - 70), "Blue: Adjacent Equality Graph", fill = (0, 0, 0, 255), font = font3)

        elif frame[4] == MATCHING_STAGE:
            d.text((2, img.height - 100), f"Augmenting Path to [Row, Column] {frame[5]}", fill = (0, 0, 0, 255), font = font3)

        elif frame[4] == FLIPPING_STAGE:
            d.text((2, img.height - 100), "Invert Matches on Red-Red Path", fill = (0, 0, 0, 255), font = font3)

        elif frame[4] == RESET_STAGE:
            d.text((2, img.height - 100), "New Initial Stage: pick a free row", fill = (0, 0, 0, 255), font = font3)

        elif frame[4] == EXPANSION_STAGE:
            d.text((2, img.height - 100), "Column was not Free: Expand Tree", fill = (0, 0, 0, 255), font = font3)

        elif frame[4] == EXIT_STAGE:
            d.text((2, img.height - 100), "Matching Full: Algorithm is Complete", fill = (0, 0, 255, 255), font = font3)

        if frame[4] == NEIGHBOUR_STAGE:
            for nb in frame[5]:
                up = (nb[0] + 1) * singular_height
                left = wide + (nb[1]) * singular_width
                d.rectangle((left, up, left + singular_width, up + singular_height), fill = (123, 226, 237, 255), outline = (123, 226, 237, 255))

        if frame[4] == UPDATE_STAGE:
            nb = frame[8]
            up = (nb[0] + 1) * singular_height
            left = wide + (nb[1]) * singular_width
            d.rectangle((left, up, left + singular_width, up + singular_height), fill = (207, 123, 237, 255), outline = (207, 123, 237, 255))

        # Shade the optimal matching
        matching = frame[3]
        for match in matching:
            up = (match[0] + 1) * singular_height
            left = wide + (match[1]) * singular_width
            d.rectangle((left, up, left + singular_width, up + singular_height), fill = (128, 128, 128, 255), outline = (128, 128, 128, 255))

        # Draw the matrix frame
        for i in range(len(frame[0]) + 1):
            d.line((wide, (i+1) * singular_height, matrix_width + wide, (i+1) * singular_height), fill = (0, 0, 0, 255))
            d.line((wide + i * singular_width, singular_height, wide + i * singular_width, singular_height + matrix_height), fill = (0, 0, 0, 255))

        # Draw the tree
        
        redcircles = []
        circles = []

        for path in frame[2]:
            midpoint_y = (path[0] + 1.5) * singular_height
            midpoint_x = wide
            redcircles.append((midpoint_x, midpoint_y))
            last_coord = wide

            first_is_row = 1
            for i in range(len(path) - 1):
                if first_is_row:

                    midpoint_x = last_coord
                    midpoint_x2 = wide + (path[i+1] + 0.5) * singular_width
                    midpoint_y = (path[i] + 1.5) * singular_height
                    d.line((midpoint_x, midpoint_y, midpoint_x2, midpoint_y), fill = (0, 128, 128, 255), width = 3)
                    circles.append((midpoint_x2, midpoint_y))

                    last_coord = midpoint_y
  
                else:

                    midpoint_y = last_coord
                    midpoint_y2 = (path[i+1] + 1.5) * singular_height
                    midpoint_x = wide + (path[i] + 0.5) * singular_width
 
                    d.line((midpoint_x, midpoint_y, midpoint_x, midpoint_y2), fill = (0, 128, 128, 255), width = 3)
                    circles.append((midpoint_x, midpoint_y2))

                    last_coord = midpoint_x

                first_is_row = 1 - first_is_row

        if frame[4] in [MATCHING_STAGE, FLIPPING_STAGE]:
            pathpair = frame[5]
            for match in frame[6]:
                if match[0] == pathpair[0] and match[1] != pathpair[1]:
                    break
            else:
                match = [0, -0.5]

            midpoint_x = wide + (match[1] + 0.5) * singular_width
            midpoint_x2 = wide + (pathpair[1] + 0.5) * singular_width
            midpoint_y = (pathpair[0] + 1.5) * singular_height
            d.line((midpoint_x, midpoint_y, midpoint_x2, midpoint_y), fill = (0, 128, 128, 255), width = 3)
            d.line((midpoint_x2, midpoint_y, midpoint_x2, singular_height), fill = (0, 128, 128, 255), width = 3)
            circles.append((midpoint_x2, midpoint_y))
            redcircles.append((midpoint_x2, singular_height))
              

        # draw circles
        scl = 5
        for circle in circles:
            midpoint_x, midpoint_y = circle
            col = (0, 0, 255, 255)
            d.ellipse([(midpoint_x - scl, midpoint_y - scl), (midpoint_x + scl, midpoint_y + scl)], fill=col, outline=col)

        for circle in redcircles:
            midpoint_x, midpoint_y = circle
            col = (255, 0, 0, 255)
            d.ellipse([(midpoint_x - scl, midpoint_y - scl), (midpoint_x + scl, midpoint_y + scl)], fill=col, outline=col)
          
            
        # Draw the numbers
        for i in range(len(frame[0])):
            if frame[4] == UPDATE_STAGE and i in frame[6]:
                col = (35, 217, 74, 255)
            else:
                col = (0, 0, 0, 255)
            d.text((20, (i+1) * singular_height + 10), str(frame[0][i]), fill = col, font = font3)

            if frame[4] == UPDATE_STAGE and i in frame[7]:
                col = (35, 217, 74, 255)
            else:
                col = (0, 0, 0, 255)
            d.text((wide + i * singular_width + 20, 10), str(frame[1][i]), fill = col, font = font3)

            for j in range(len(frame[0])):
                d.text((wide + j * singular_width, (i+1) * singular_height), str(matrix[i, j]), fill = (0, 0, 0, 255), font = font3)
        
        output_images.append(img)

    return output_images

def get_paths(nodes):
    """Computes all alternating path subsets of an alternating tree. May contain overlaps."""

    all_paths = []
    for node in nodes:
        cur = nodes[node]
        path = [cur.val]
        while cur.parent != None:
            cur = cur.parent
            path.insert(0, cur.val)

        all_paths.insert(0, path)

    return all_paths


def hungarian(matrx):
    """Runs the Hungarian Algorithm on a given matrix and returns the optimal matching with potentials. Produces intermediate images while executing."""
    
    frames = []

    # Step 1: Prep matrix, get size
    matrx = np.array(matrx)
    size = matrx.shape[0]
    
    # Step 2: Generate trivial potentials
    rpotentials = []
    cpotentials = [0 for i in range(size)]
    for i in range(len(matrx)):
        row = matrx[i]
        rpotentials.append(max(row))


    # Step 3: Initialize alternating tree
    matching = []
    S = {0}
    T = set()

    tree_root = Node(0)
    x_nodes = {0: tree_root}

    frames.append([rpotentials.copy(), cpotentials.copy(), get_paths(x_nodes), matching.copy(), INITIAL_STAGE])

    # Create helper functions

    def neighbours(wset):
        """Finds all firms in equality graph with workers in wset."""
    
        result = []
        for x in wset:
            # get row of firms for worker x
            nbs = matrx[x, :]
            for y in range(len(nbs)):
                # check for equality
                if nbs[y] == rpotentials[x] + cpotentials[y]:
                    result.append([x, y])

        return result
    

    def update_potentials():
        """Find the smallest difference between treed workers and untreed firms 
            and use it to update potentials."""
        
        # when using functions in functions, if modifying variables, call nonlocal
        nonlocal rpotentials, cpotentials 
        big = np.inf
        args = None
        # iterate over relevant pairs
        for dx in S:
            for dy in set(range(size)) - T:
                # find the difference and check if its smaller than any we found before
                weight = matrx[dx, dy]
                alpha = rpotentials[dx] + cpotentials[dy] - weight
                if alpha < big:
                    big = alpha
                    args = [dx, dy]

        # apply difference to potentials as needed
        for dx in S:
            rpotentials[dx] -= big

        for dy in T:
            cpotentials[dy] += big

        return big, S, T, args
        
    # Step 4: Loop while our matching is too small
    while len(matching) != size:
        # Step A: Compute neighbours in equality graph
        NS = neighbours(S)
        frames.append([rpotentials.copy(), cpotentials.copy(), get_paths(x_nodes), matching.copy(), NEIGHBOUR_STAGE, NS])
        if set([b[1] for b in NS]) == T:
            # Step B: If all firms are in the tree, update potentials to get a new one
            alpha, ds, dt, args = update_potentials()
            NS = neighbours(S)
            frames.append([rpotentials.copy(), cpotentials.copy(), get_paths(x_nodes), matching.copy(), UPDATE_STAGE, alpha, ds.copy(), dt.copy(), args])
            frames.append([rpotentials.copy(), cpotentials.copy(), get_paths(x_nodes), matching.copy(), NEIGHBOUR_STAGE, NS])

        # get the untreed firm
        pair = next(n for n in NS if n[1] not in T)
        if pair[1] not in [m[1] for m in matching]:
            # Step D: Firm is not matched so add it to matching 
            thecopy = matching.copy()
            frames.append([rpotentials.copy(), cpotentials.copy(), get_paths(x_nodes), thecopy, MATCHING_STAGE, pair, thecopy])
            matching.append(pair)
            # Step E: Swap the alternating path in our alternating tree attached to the worker we matched
            source = x_nodes[pair[0]]
            matched = 1
            while source.parent != None:
                above = source.parent
                if matched:
                    # if previously matched, this should be removed from matching
                    matching.remove([source.val, above.val])
                else:
                    # if previous was a remove, this is a match
                    matching.append([above.val, source.val])

                matched = 1 - matched
                source = above

            frames.append([rpotentials.copy(), cpotentials.copy(), get_paths(x_nodes), matching.copy(), FLIPPING_STAGE, pair, thecopy])

            # Step F: Destroy the tree, go to Step 4 to check completion, and possibly go to Step A
            free = list(set(range(size)) - set([m[0] for m in matching]))
            if len(free):
                tree_root = Node(free[0])
                x_nodes = {free[0]: tree_root}
                S = {free[0]}
                T = set()
                frames.append([rpotentials.copy(), cpotentials.copy(),get_paths(x_nodes), matching.copy(), RESET_STAGE])

            else:
                x_nodes = {}
                S = set()
                T = set()  

        else:
            # Step C: Firm is matched so add it to the tree and go back to Step A
            matching_x = next(m[0] for m in matching if m[1] == pair[1])
            S.add(matching_x)
            T.add(pair[1])
            source = x_nodes[pair[0]]
            y_node = Node(pair[1], source)
            x_node = Node(matching_x, y_node)
            x_nodes[matching_x] = x_node

            frames.append([rpotentials.copy(), cpotentials.copy(), get_paths(x_nodes), matching.copy(), EXPANSION_STAGE])
    
    revenues = [matrx[m[0], m[1]] for m in matching]
    class Result:
        """A simple response object."""

        def __init__(self, match, revenues, row_weights, col_weights, revenue_sum, result, matrix):
            self.match = match
            self.revenues = revenues
            self.row_weights = row_weights
            self.col_weights = col_weights
            self.revenue_sum = revenue_sum
            self.frames = process_frames(result, matrix)

        def __str__(self):
            size = len(self.match)
            maxlen = max(len(str(max(self.revenues))), len(str(min(self.revenues))))
            baselist = [[" "*maxlen for i in range(size)] for j in range(size)]
            for i in range(size):
                entry = self.match[i]
                baselist[entry[0]][entry[1]] = str(self.revenues[i]).rjust(maxlen)

            formatted_list = '\n'.join([str(row) for row in baselist])
            return f"Matching:\n{formatted_list}\n\nRow Potentials: {self.row_weights}\nColumn Potentials: {self.col_weights}"

    frames.append([rpotentials.copy(), cpotentials.copy(), get_paths(x_nodes), matching.copy(), EXIT_STAGE])

    return Result(matching, revenues, rpotentials, cpotentials, sum(revenues), frames, matrx)