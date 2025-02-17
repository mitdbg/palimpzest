"""

@kylel

"""

import json
import os
import pathlib

from papermage.magelib import Document
from papermage.recipes import CoreRecipe
from papermage.visualizers.visualizer import plot_entities_on_page

# load doc
recipe = CoreRecipe()
# pdfpath = pathlib.Path(__file__).parent.parent / "tests/fixtures/1903.10676.pdf"
# pdfpath = pathlib.Path(__file__).parent.parent / "tests/fixtures/2304.02623v1.pdf"
pdfpath = pathlib.Path(__file__).parent.parent / "tests/fixtures/2020.acl-main.447.pdf"
# pdfpath = pathlib.Path(__file__).parent.parent / "tests/fixtures/2305.14772.pdf"
doc = recipe.from_pdf(pdf=pdfpath)

# visualize tokens
page_id = 0
plot_entities_on_page(page_image=doc.images[page_id], entities=doc.pages[page_id].tokens)

# visualize tables
page_id = 5
tables = doc.pages[page_id].intersect_by_box("tables")
plot_entities_on_page(page_image=doc.images[page_id], entities=tables)
for table in tables:
    print(table.text)

# visualize figures
figures = doc.pages[page_id].intersect_by_box("figures")
for figure in figures:
    print(figure.text)
plot_entities_on_page(page_image=doc.images[page_id], entities=figures)

# visualize blocks
blocks = doc.pages[page_id].intersect_by_box("blocks")
for block in blocks:
    print(block.text)
plot_entities_on_page(page_image=doc.images[page_id], entities=blocks)

# visualize sections
page_id = 4
sections = doc.pages[page_id].intersect_by_box("sections")
for section in sections:
    print(section.text)
plot_entities_on_page(page_image=doc.images[page_id], entities=sections)


# get the image of a page and its dimensions
page_image = doc.images[page_id]
page_w, page_h = page_image.pilimage.size

# get the bounding box of a figure
figure_box = figures[0].boxes[0]

# convert it
figure_box_xy = figure_box.to_absolute(page_width=page_w, page_height=page_h).xy_coordinates

# crop the image using PIL
page_image._pilimage.crop(figure_box_xy)


##### BUG

# weird.  this says no figures on page 8, but there are
doc.pages[8].intersect_by_box("figures")
page = doc.pages[8]

# get figures on 8th page.
figures = [figure for figure in doc.figures if figure.boxes[0].page == 8]
assert len(figures) > 0
print(f"{figures[0].boxes}")

# weird.. it's on 8th page
print(f"{figure.boxes[0].xy_coordinates}")

plot_entities_on_page(page_image=doc.images[8], entities=figures)


# but yet it doesnt intersect pages
figure.intersect_by_box("pages")

print(f"{page.boxes[0].xy_coordinates}")

figure.boxes[0].is_overlap(page.boxes[0])
