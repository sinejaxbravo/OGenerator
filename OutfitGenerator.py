import asyncio
from Weather import extract_weather
import Directories as dir
from main import make_pairs

sesh = asyncio.new_event_loop()
temperature, precipitation, overcast = sesh.run_until_complete(extract_weather())
# print(temperature)
# print(temperature)

paths = dir.clothing_folders

paths_to_use = [paths["pant"], paths["shirt"], paths["coat"], paths["shoe"]]
names = ["pant", "shirt", "coat", "shoe"]
output_path_to_use = paths["pair"]

pant_shirt = make_pairs([paths["pant"], paths["shirt"]], ["pant", "shirt"], output_path_to_use)
print(pant_shirt)

# pant_shoe = make_pairs([paths["pant"], paths["shoe"]], ["pant", "shoe"], output_path_to_use)
# pant_coat = make_pairs([paths["pant"], paths["coat"]], ["pant", "coat"], output_path_to_use)
# shoe_shirt = make_pairs([paths["shoe"], paths["shirt"]], ["shoe", "shirt"], output_path_to_use)
# shoe_coat = make_pairs([paths["shoe"], paths["coat"]], ["shoe", "coat"], output_path_to_use)
# coat_shirt = make_pairs([paths["coat"], paths["shirt"]], ["coat", "shirt"], output_path_to_use)

with_coat = []
without_coat = []


