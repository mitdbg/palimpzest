import argparse
import json
import os
import time

import gradio as gr
import numpy as np
from PIL import Image
import pandas as pd
import requests
import urllib
from typing import List, Dict, Any

import palimpzest as pz
from palimpzest.core.lib.fields import ImageFilepathField, ListField
from palimpzest.utils.udfs import xls_to_tables

# Addresses far from MIT; we use a simple lookup like this to make the
# experiments re-producible w/out needed a Google API key for geocoding lookups
FAR_AWAY_ADDRS = [
    "Melcher St",
    "Sleeper St",
    "437 D St",
    "Seaport Blvd",
    "50 Liberty Dr",
    "Telegraph St",
    "Columbia Rd",
    "E 6th St",
    "E 7th St",
    "E 5th St",
]


def within_two_miles_of_mit(record: dict):
    # NOTE: I'm using this hard-coded function so that folks w/out a
    #       Geocoding API key from google can still run this example
    try:
        return not any([street.lower() in record["address"].lower() for street in FAR_AWAY_ADDRS])
    except Exception:
        return False


def in_price_range(record: dict):
    try:
        price = record["price"]
        if isinstance(price, str):
            price = price.strip()
            price = int(price.replace("$", "").replace(",", ""))
        return 6e5 < price <= 2e6
    except Exception:
        return False

email_cols =  [
    {"name": "sender", "type": str, "desc": "The email address of the sender"},
    {"name": "subject", "type": str, "desc": "The subject of the email"},
]

case_data_cols = [
    {"name": "case_submitter_id", "type": str, "desc": "The ID of the case"},
    {"name": "age_at_diagnosis", "type": int | float, "desc": "The age of the patient at the time of diagnosis"},
    {"name": "race", "type": str, "desc": "An arbitrary classification of a taxonomic group that is a division of a species."},
    {"name": "ethnicity", "type": str, "desc": "Whether an individual describes themselves as Hispanic or Latino or not."},
    {"name": "gender", "type": str, "desc": "Text designations that identify gender."},
    {"name": "vital_status", "type": str, "desc": "The vital status of the patient"},
    {"name": "ajcc_pathologic_t", "type": str, "desc": "Code of pathological T (primary tumor) to define the size or contiguous extension of the primary tumor (T), using staging criteria from the American Joint Committee on Cancer (AJCC)."},
    {"name": "ajcc_pathologic_n", "type": str, "desc": "The codes that represent the stage of cancer based on the nodes present (N stage) according to criteria based on multiple editions of the AJCC's Cancer Staging Manual."},
    {"name": "ajcc_pathologic_stage", "type": str, "desc": "The extent of a cancer, especially whether the disease has spread from the original site to other parts of the body based on AJCC staging criteria."},
    {"name": "tumor_grade", "type": int | float, "desc": "Numeric value to express the degree of abnormality of cancer cells, a measure of differentiation and aggressiveness."},
    {"name": "tumor_focality", "type": str, "desc": "The text term used to describe whether the patient's disease originated in a single location or multiple locations."},
    {"name": "tumor_largest_dimension_diameter", "type": int | float, "desc": "The tumor largest dimension diameter."},
    {"name": "primary_diagnosis", "type": str, "desc": "Text term used to describe the patient's histologic diagnosis, as described by the World Health Organization's (WHO) International Classification of Diseases for Oncology (ICD-O)."},
    {"name": "morphology", "type": str, "desc": "The Morphological code of the tumor, as described by the World Health Organization's (WHO) International Classification of Diseases for Oncology (ICD-O)."},
    {"name": "tissue_or_organ_of_origin", "type": str, "desc": "The text term used to describe the anatomic site of origin, of the patient's malignant disease, as described by the World Health Organization's (WHO) International Classification of Diseases for Oncology (ICD-O)."},
    {"name": "study", "type": str, "desc": "The last name of the author of the study, from the table name"},
    {"name": "filename", "type": str, "desc": "The name of the file the record was extracted from"}
]

real_estate_listing_cols = [
    {"name": "listing", "type": str, "desc": "The name of the listing"},
    {"name": "text_content", "type": str, "desc": "The content of the listing's text description"},
    {"name": "image_filepaths", "type": ListField(ImageFilepathField), "desc": "A list of the filepaths for each image of the listing"},
]

real_estate_text_cols = [
    {"name": "address", "type": str, "desc": "The address of the property"},
    {"name": "price", "type": int | float, "desc": "The listed price of the property"},
]

real_estate_image_cols = [
    {"name": "is_modern_and_attractive", "type": bool, "desc": "True if the home interior design is modern and attractive and False otherwise"},
    {"name": "has_natural_sunlight", "type": bool, "desc": "True if the home interior has lots of natural sunlight and False otherwise"},
]

table_cols = [
    {"name": "rows", "type": list[str], "desc": "The rows of the table"},
    {"name": "header", "type": list[str], "desc": "The header of the table"},
    {"name": "name", "type": str, "desc": "The name of the table"},
    {"name": "filename", "type": str, "desc": "The name of the file the table was extracted from"}
]

music_knowledge_graph_cols = [
    {"name": "record_type", "type": str, "desc": "The type of record - artist or song"},
    {"name": "artist_name", "type": str, "desc": "The name of the artist"},
    {"name": "artist_wikipedia_summary", "type": str, "desc": "The wikipedia summary of the artist"},
    {"name": "song_title", "type": str, "desc": "The title of the song"},
    {"name": "song_length_ms", "type": int | float, "desc": "The length of the song in milliseconds"},
    {"name": "song_artist_names", "type": list[str], "desc": "The names of the artists who collaborated on the song"}
]

# class RealEstateListingFiles(Schema):
#     """The source text and image data for a real estate listing."""

#     listing = StringField(desc="The name of the listing")
#     text_content = StringField(desc="The content of the listing's text description")
#     image_filepaths = ListField(
#         element_type=ImageFilepathField,
#         desc="A list of the filepaths for each image of the listing",
#     )

class RealEstateListingReader(pz.DataReader):
    def __init__(self, listings_dir):
        super().__init__(real_estate_listing_cols)
        self.listings_dir = listings_dir
        self.listings = sorted(os.listdir(self.listings_dir))

    def __len__(self):
        return len(self.listings)

    def __getitem__(self, idx: int):
        # get listing
        listing = self.listings[idx]

        # get fields
        image_filepaths, text_content = [], None
        listing_dir = os.path.join(self.listings_dir, listing)
        for file in os.listdir(listing_dir):
            if file.endswith(".txt"):
                with open(os.path.join(listing_dir, file), "rb") as f:
                    text_content = f.read().decode("utf-8")
            elif file.endswith(".png"):
                image_filepaths.append(os.path.join(listing_dir, file))

        # construct and return dictionary with fields
        return {"listing": listing, "text_content": text_content, "image_filepaths": image_filepaths}
    


def api_request(url, params=None, headers={"User-Agent": "pz/0.1 (contact: minecraftandcomputers@gmail.com)"}, max_retries=3, session=None):
    """
    A wrapper for making API requests with:
    - Automatic retries (handles 503 errors)
    - Rate limit handling (respects 'Retry-After' header)
    - Persistent session support for efficiency
    """
    session = session or requests.Session()  # Use provided session or create a new one
    
    for attempt in range(max_retries):
        resp = session.get(url, params=params, headers=headers)

        if resp.status_code == 200:
            return resp.json()
        
        elif resp.status_code == 503:
            retry_after = int(resp.headers.get("Retry-After", 5))  # Default to 5 seconds
            print(f"503 error. Retrying in {retry_after} seconds...")
            time.sleep(retry_after)
        else:
            resp.raise_for_status()
    
    raise requests.exceptions.HTTPError(f"Failed after {max_retries} retries.")


class MusicKnowledgeGraphReader(pz.DataReader):
    def __init__(self, artist_list: List[str]):
        """Initialize and build a music knowledge graph for the given artists.
        
        Args:
            artist_list: List of artist names to fetch data for
        """
        super().__init__(music_knowledge_graph_cols)
        self.artist_list = artist_list
        self.session = requests.Session()
        
        # Fetch all data
        self.data = self._build_knowledge_graph()
        # print the entries where record_type is artist
        print(self.data[self.data["record_type"] == "artist"])

    def _build_knowledge_graph(self) -> pd.DataFrame:
        """Build the complete knowledge graph by fetching all required data."""
        all_rows = []
        
        # Batch fetch artist data and Wikipedia summaries
        artist_data = self._fetch_multiple_artists(self.artist_list)
        artist_names = [artist["name"] for artist in artist_data if "name" in artist]
        wikipedia_summaries = self._fetch_wikipedia_summaries(artist_names)

        # Process each artist and their recordings
        for artist, wiki_summary in zip(artist_data, wikipedia_summaries):
            artist_name = artist.get("name", "Unknown Artist")
            artist_id = artist.get("id")
            
            if not artist_id:
                continue

            # Add artist record
            all_rows.append({
                "record_type": "artist",
                "artist_name": artist_name,
                "artist_wikipedia_summary": wiki_summary,
                "song_title": "",
                "song_length_ms": None,
                "song_artist_names": []
            })
            
            # Fetch and add song records
            recordings = self._fetch_recordings_for_artist(artist_id)
            for recording in recordings:
                credited_names = [
                    credit["artist"]["name"]
                    for credit in recording.get("artist-credit", [])
                    if isinstance(credit, dict) and "artist" in credit
                ]

                all_rows.append({
                    "record_type": "song",
                    "artist_name": artist_name,
                    "artist_wikipedia_summary": "",
                    "song_title": recording.get("title", "Unknown Title"),
                    "song_length_ms": recording.get("length"),
                    "song_artist_names": credited_names
                })

        return pd.DataFrame(all_rows, columns=[
            "record_type",
            "artist_name",
            "artist_wikipedia_summary",
            "song_title",
            "song_length_ms",
            "song_artist_names"
        ])

    def _fetch_multiple_artists(self, names: List[str]) -> List[Dict[str, Any]]:
        """Fetch multiple artists from MusicBrainz API."""
        artists = []
        
        for name in names:
            try:
                params = {
                    "query": f"artist:{name}",
                    "fmt": "json",
                    "limit": 1
                }
                data = api_request(
                    "https://musicbrainz.org/ws/2/artist",
                    params=params,
                    session=self.session
                )
                results = data.get("artists", [])
                if results:
                    artists.append(results[0])
                
            except requests.RequestException as e:
                print(f"Error fetching artist {name}: {e}")
                continue
                
        return artists

    def _fetch_recordings_for_artist(self, artist_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Fetch recordings for a given MusicBrainz artist ID."""
        try:
            params = {
                "artist": artist_id,
                "fmt": "json",
                "limit": limit
            }
            data = api_request(
                "https://musicbrainz.org/ws/2/recording",
                params=params,
                session=self.session
            )
            return data.get("recordings", [])
            
        except requests.RequestException as e:
            print(f"Error fetching recordings for artist {artist_id}: {e}")
            return []

    def _fetch_wikipedia_summaries(self, artist_names: List[str]) -> List[str]:
        """Fetch Wikipedia summaries for multiple artists."""
        summaries = []
        
        for artist_name in artist_names:
            try:
                # First, search for the Wikipedia page
                search_params = {
                    "action": "query",
                    "list": "search",
                    "srsearch": f"{artist_name} musician",  # Add 'musician' for better results
                    "format": "json",
                    "srlimit": 1
                }
                
                search_data = api_request(
                    "https://en.wikipedia.org/w/api.php",
                    params=search_params,
                    session=self.session
                )
                
                search_results = search_data.get("query", {}).get("search", [])
                if not search_results:
                    summaries.append("")
                    continue

                # Then fetch the summary
                page_title = search_results[0]["title"]
                encoded_title = urllib.parse.quote(page_title)
                summary_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{encoded_title}"
                
                summary_data = api_request(
                    summary_url,
                    session=self.session
                )
                summaries.append(summary_data.get("extract", ""))
                
            except requests.RequestException as e:
                print(f"Error fetching Wikipedia summary for {artist_name}: {e}")
                summaries.append("")
                continue
                
        return summaries
    
    def __repr__(self):
        return f"DataReader containing {len(self.data)} rows."

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        # return index from self.data dataframe
        return self.data.iloc[idx]


def main():

    # parse arguments
    parser = argparse.ArgumentParser(description="Run a simple demo")
    parser.add_argument("--viz", default=False, action="store_true", help="Visualize output in Gradio")
    parser.add_argument("--verbose", default=False, action="store_true", help="Print verbose output")
    parser.add_argument("--profile", default=False, action="store_true", help="Profile execution")
    parser.add_argument("--dataset", type=str, help="The path to the dataset")
    parser.add_argument(
        "--workload", type=str, help="The workload to run. One of enron, real-estate, medical-schema-matching."
    )
    parser.add_argument(
        "--executor",
        type=str,
        help="The plan executor to use. One of sequential, pipelined, parallel",
        default="sequential",
    )
    parser.add_argument(
        "--policy",
        type=str,
        help="One of 'mincost', 'mintime', 'maxquality'",
        default="mincost",
    )

    args = parser.parse_args()

    # The user has to indicate the workload and the datasetid if the workload is not music-knowledge-graph
    if args.dataset is None and args.workload != "music-knowledge-graph":
        print("Please provide a dataset id")
        exit(1)
    if args.workload is None:
        print("Please provide a workload")
        exit(1)

    # create directory for profiling data
    if args.profile:
        os.makedirs("profiling-data", exist_ok=True)

    dataset = args.dataset
    workload = args.workload
    visualize = args.viz
    verbose = args.verbose
    profile = args.profile
    policy = pz.MaxQuality()
    if args.policy == "mincost":
        policy = pz.MinCost()
    elif args.policy == "mintime":
        policy = pz.MinTime()
    elif args.policy == "maxquality":
        policy = pz.MaxQuality()
    else:
        print("Policy not supported for this demo")
        exit(1)

    if os.getenv("OPENAI_API_KEY") is None and os.getenv("TOGETHER_API_KEY") is None:
        print("WARNING: Both OPENAI_API_KEY and TOGETHER_API_KEY are unset")

    # create pz plan
    if workload == "enron":
        plan = pz.Dataset(dataset).sem_add_columns(email_cols)
        plan = plan.sem_filter(
            "The email is not quoting from a news article or an article written by someone outside of Enron"
        )
        plan = plan.sem_filter(
            'The email refers to a fraudulent scheme (i.e., "Raptor", "Deathstar", "Chewco", and/or "Fat Boy")'
        )

    elif workload == "real-estate":
        plan = pz.Dataset(RealEstateListingReader(dataset))
        plan = plan.sem_add_columns(real_estate_text_cols, depends_on="text_content")
        plan = plan.sem_add_columns(real_estate_image_cols, depends_on="image_filepaths")
        plan = plan.sem_filter(
            "The interior is modern and attractive, and has lots of natural sunlight",
            depends_on=["is_modern_and_attractive", "has_natural_sunlight"],
        )
        plan = plan.filter(within_two_miles_of_mit, depends_on="address")
        plan = plan.filter(in_price_range, depends_on="price")

    elif workload == "medical-schema-matching":
        plan = pz.Dataset(dataset)
        plan = plan.add_columns(xls_to_tables, cols=table_cols, cardinality=pz.Cardinality.ONE_TO_MANY)
        plan = plan.sem_filter("The rows of the table contain the patient age")
        plan = plan.sem_add_columns(case_data_cols, cardinality=pz.Cardinality.ONE_TO_MANY)

    elif workload == "music-knowledge-graph":
        # Name of artists who worked w/artist X, and who meets some condition
        plan = pz.Dataset(MusicKnowledgeGraphReader(["Drake", "Travis Scott", "Kendrick Lamar", "Playboi Carti", "Ed Sheeran", "Taylor Swift", "Justin Bieber", "Selena Gomez", "Megan Thee Stallion"]))

        plan = plan.sem_filter("The row represents an artist who's won more than 5 Grammy awards", depends_on=["record_type", "artist_wikipedia_summary"])
        
        # # extract the list of the remaining artists
        # plan = plan.project(["artist_name"])

        # # use these artists to go through the row records and find the songs where an artist from this list collaborated with Travis Scott
        # plan2 = plan


    # construct config and run plan
    config = pz.QueryProcessorConfig(
        cache=False,
        verbose=verbose,
        policy=policy,
        execution_strategy=args.executor,
    )
    data_record_collection = plan.run(config)
    print(data_record_collection.to_df())

    # save statistics
    if profile:
        stats_path = f"profiling-data/{workload}-profiling.json"
        execution_stats_dict = data_record_collection.execution_stats.to_json()
        with open(stats_path, "w") as f:
            json.dump(execution_stats_dict, f)

    # visualize output in Gradio
    if visualize:
        from palimpzest.utils.demo_helpers import print_table

        plan_str = list(data_record_collection.execution_stats.plan_strs.values())[-1]
        if workload == "enron":
            print_table(data_record_collection.data_records, cols=["sender", "subject"], plan_str=plan_str)

        elif workload == "real-estate":
            fst_imgs, snd_imgs, thrd_imgs, addrs, prices = [], [], [], [], []
            for record in data_record_collection:
                addrs.append(record.address)
                prices.append(record.price)
                for idx, img_name in enumerate(["img1.png", "img2.png", "img3.png"]):
                    path = os.path.join(dataset, record.listing, img_name)
                    img = Image.open(path)
                    img_arr = np.asarray(img)
                    if idx == 0:
                        fst_imgs.append(img_arr)
                    elif idx == 1:
                        snd_imgs.append(img_arr)
                    elif idx == 2:
                        thrd_imgs.append(img_arr)

            with gr.Blocks() as demo:
                fst_img_blocks, snd_img_blocks, thrd_img_blocks, addr_blocks, price_blocks = [], [], [], [], []
                for fst_img, snd_img, thrd_img, addr, price in zip(fst_imgs, snd_imgs, thrd_imgs, addrs, prices):
                    with gr.Row(equal_height=True):
                        with gr.Column():
                            fst_img_blocks.append(gr.Image(value=fst_img))
                        with gr.Column():
                            snd_img_blocks.append(gr.Image(value=snd_img))
                        with gr.Column():
                            thrd_img_blocks.append(gr.Image(value=thrd_img))
                    with gr.Row():
                        with gr.Column():
                            addr_blocks.append(gr.Textbox(value=addr, info="Address"))
                        with gr.Column():
                            price_blocks.append(gr.Textbox(value=price, info="Price"))

                plan_str = list(data_record_collection.execution_stats.plan_strs.values())[0]
                gr.Textbox(value=plan_str, info="Query Plan")

            demo.launch()

if __name__ == "__main__":

    main()

