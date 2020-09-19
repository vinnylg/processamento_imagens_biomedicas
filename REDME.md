CLASSES:

0 - non-lung
1 - normal
2 - emphysema
3 - ground-glass
4 - fibrosis
5 - micronodules
6 - consolidation


FILENAME CONVENTION:

{img_name}_{roi_number}_{coord_x}-{coord_y}_{percent}_{label}.dcm
img_name = original image name
roi_number = number of the roi found in the mask
coord_x = x coordinate where starts the ROI rectangle
coord_y = y coordinate where starts the ROI rectangle
percent = percentage of pixels falling inside the roi
label = label of the roi

