import fitz
import sys
import os
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pickle
from shutil import rmtree

def pdf2images(pdf_fn):
    output_folder_path = 'tmp'
    if not os.path.exists(output_folder_path):
        os.mkdir(output_folder_path)

    pdffile = sys.argv[1]
    doc = fitz.open(pdffile)
    for i in range(doc.pageCount):
        page = doc.loadPage(i)
        # Use 2x matrix to increase resolution
        mat = fitz.Matrix(2., 2.)
        pix = page.getPixmap(mat)
        output_path = os.path.join(output_folder_path, f"{i+1}.png")
        pix.writePNG(output_path)

    return doc.pageCount

def _display(imgs, resize=None):
    while True:
        for i, img in enumerate(imgs):
            if resize:
                img = cv2.resize(img, None, fx=resize, fy=resize)
            cv2.imshow(f"Display {i}", img)

        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break

    cv2.destroyAllWindows()

def checkDistance(pt1, pt2):
    ''' Computes the euclidean distance between two points '''
    pt1 = np.array(pt1)
    pt2 = np.array(pt2)
    return np.sqrt(np.sum(np.square(pt2 - pt1)))

class GraphExtractor:
    REFERENCES_PATH = "tm_refs/"
    REFERENCES = ["v_scale1.png", "v_scale2.png"]
    TM_THRESHOLD = 0.7

    # HSV filter for blue color
    BLUE_LOWER = np.array([100, 60, 60])
    BLUE_UPPER = np.array([120, 255, 255])

    # Categories in the order they appear
    CATEGORIES = ["RR", # Retail and Recreation
                  "GP", # Grocery and pharmacy
                  "Pa", # Parks
                  "TS", # Transit stations
                  "Wo", # Workplace
                  "Re"] # Residential

    # Regions and the order they appear
    REGIONS = ["Bra", # Whole of brazil
               "DF", "AC", "AL", "AP", "AM", "BA", "CE", "ES", "GO",
               "MA", "MT", "MS", "MG", "PR", "PB", "PA", "PE", "PI", "RN", "RS",
               "RJ", "RO", "RR", "SC", "SE", "SP", "TO"]

    def __init__(self):
        self._loadReferences()

    def _loadReferences(self):
        ''' Load the reference images used for template matching '''
        self.ref_imgs = {}

        self.ref_imgs["ref1"] = cv2.imread(
                os.path.join(self.REFERENCES_PATH, self.REFERENCES[0]))

        self.ref_imgs["ref2"] = cv2.imread(
                os.path.join(self.REFERENCES_PATH, self.REFERENCES[1]))

    def _digitizeGraph(self, img, anchor_coords, ref_dim, 
            returnNonQuantized=False):
        ''' Receives an anchor coordinate pointing to the topmost leftmost
        coordinate just before (one pixel off, ideally) from the start of the
        graph. Incrementally checks each following column, finding the blue
        pixels. The Y coordinate is determined by the average of the y
        coordinates where blue pixels were found. Normalizes the Y value
        by the height of the reference image used for template matching.
        The topmost pixel of the template corresponds exactly to the +80%
        mark of the graph, and the bottommost pixel to -80%.

        '''
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Using twice the size of the reference because some data might be off
        # the graph
        height = int(ref_dim[0] * 2)
        y = anchor_coords[1]
        start_y = int(y - height / 4.)
        end_y = int(start_y + height)

        # TODO: Make this a function argument
        days = 43

        # The zero mark, in pixels. Used to normalize the values
        zero = height / 2.
        # Used to normalize pixel to percentage. The scale ranges from 0 to 80%
        # This determines how many pixels the scale would be if it ranged to 
        # 100%
        hundred_per_h = (zero/2) / 0.8
        
        # Start column
        col = anchor_coords[0] + ref_dim[1]

        #roi = img[start_y:end_y, col-50:col+50]
        #_display([roi])
        
        stopFlag = False
        skipCount = 0
        inc = 0
        data_buffer = []
        while not stopFlag:
            # Extract next col
            currCol = hsv[start_y:end_y, col+inc].reshape(height, 1, 3)
            # Mask blue pixels
            filteredCol = cv2.inRange(currCol, self.BLUE_LOWER, self.BLUE_UPPER)
            # Find indices where the blue pixels occured
            matchIndices = np.argwhere(filteredCol > 0)

            # Check stop condition, i.e. 5 columns missing pixels in a row
            if matchIndices.shape[0] == 0:
                #print(f"Skip on col: {inc}.")
                skipCount += 1
                inc += 1
                if skipCount == 5:
                    stopFlag = True
                continue
            else:
                skipCount = 0

            # Find the average Y of where the blue pixels occurred
            avgY = np.average(matchIndices[:, 0])
            # Normalize the value to a percentage, positive or negative, from
            # the baseline (middle of the graph)
            normY = (zero - avgY) / hundred_per_h

            # Append data
            data_buffer.append(normY)

            inc += 1

        # Convert data to numpy array
        data_buffer = np.array(data_buffer)

        # Quantize the data along the X axis, by sampling in the frequency 
        # needed to match number of days
        quantized_buffer = []
        total_points = len(data_buffer)
        for i in range(days):
            ind = int((i / days) * total_points)
            quantized_buffer.append(data_buffer[ind])
        quantized_buffer = np.array(quantized_buffer)

        #plt.plot(np.arange(len(quantized_buffer)), quantized_buffer)
        #plt.ylim([-1.2, 1.2])
        #plt.show()

        if returnNonQuantized:
            return data_buffer, quantized_buffer
        else:
            return quantized_buffer 

    def localizeAndExtractGraphs(self, pageCount, pages_path, output_fn):
        # Keeps track of how many graphs have been saved
        graph_counter = 0
        # The data structure that will hold all graph data
        graph_dict = {}

        n_regions = len(self.REGIONS)
        n_categories = len(self.CATEGORIES)
        region_index = 0

        # Get image filenames from path
        # TODO: change back
        for page in range(1, pageCount+1):
            img_path = os.path.join(pages_path, str(page) + ".png")

            img = cv2.imread(img_path)    

            # Keeps track of already drawn points, so we don't double count any
            # graph
            drawn = []
            for ref_name, ref in self.ref_imgs.items():
                # Do template matching
                res = cv2.matchTemplate(img, ref, cv2.TM_CCOEFF_NORMED)

                # Template reference width and height
                w, h = ref.shape[:2][::-1]

                indices = np.argwhere(res > self.TM_THRESHOLD)
                for y, x in indices:
                    # Check if a close point has already been drawn
                    overlapFlag = False
                    for (m_x, m_y) in drawn:
                        if checkDistance((m_x, m_y), (x, y)) < 10:
                            overlapFlag = True
                            break

                    if not overlapFlag:
                        # Append to drawn points, so we don't repeat it
                        drawn.append((x, y))
                        
                        # Digitize the graph, get numpy array of graph data
                        graph = self._digitizeGraph(img, (x, y), ref.shape)

                        cat_index = graph_counter % n_categories 
                        region = self.REGIONS[region_index]
                        category = self.CATEGORIES[cat_index]

                        print(region, category)

                        # This is the first graph, create inner dictionary for 
                        # this region
                        if cat_index == 0:
                            graph_dict[region] = {category: graph}
                        else:
                            graph_dict[region][category] = graph

                        graph_counter += 1
                        if cat_index == n_categories-1:
                            region_index += 1

            print(f"Found {len(drawn)} graphs in page {str(page)}.")

        print(f"Dumping to file `{output_fn}`.")
        with open(output_fn, "wb") as f:
            pickle.dump(graph_dict, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extracts the graphs from\
            Google's mobility reports and converts them to numpy arrays.")
    parser.add_argument("PDF_File")
    parser.add_argument("--output", default="graph_data.pickle", 
            help="Output filename")
    parser.add_argument("--keep", action="store_true",
            help="If passed, keeps the temporary page files after the data has\
                    been extracted")

    args = parser.parse_args()
    print("Converting PDF pages to images...")
    pageCount = pdf2images(args.PDF_File)

    print("Localizing and extracting the graph data...")
    ge = GraphExtractor()
    ge.localizeAndExtractGraphs(pageCount, "tmp/", args.output)
    if not args.keep:
        rmtree("tmp/")

    print("Done.")
