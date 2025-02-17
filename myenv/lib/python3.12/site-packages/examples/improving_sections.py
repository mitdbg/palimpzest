
from papermage.recipes import CoreRecipe

recipe = CoreRecipe()
doc = recipe.run("/Users/kylel/ai2/paperMage/tests/fixtures/papermage.pdf")

# problematic section
weird_section = doc.sections[1]
print(weird_section)

# it's split between 3 rows
for row in weird_section.rows:
    print(row)

# but visually, this should all be one big row. why is it split?
from papermage.visualizers.visualizer import plot_entities_on_page

page_with_weird_section = weird_section.pages[0].id
plot_entities_on_page(page_image=doc.images[page_with_weird_section], entities=weird_section.rows)

