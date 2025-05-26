# class Bike:
#     def __init__(self):

import argparse
import json
import os
from pathlib import Path

import gradio as gr
import numpy as np
from PIL import Image
import palimpzest as pz
from palimpzest.core.data.datareaders import DataReader
from palimpzest.core.elements.records import DataRecord
from palimpzest.core.lib.fields import BooleanField, ListField, ImageFilepathField, NumericField, StringField
from palimpzest.core.lib.schemas import Schema
from palimpzest.constants import Model
from palimpzest.policy import MaxQuality, MinCost, MinTime
from palimpzest.sets import Dataset
from palimpzest.query.processor.config import QueryProcessorConfig
from palimpzest.utils.demo_helpers import print_table


import requests
import http.client




tripadvisor_location_ids = {
    "New York City": "60763",
    "Los Angeles": "32655",
    "Chicago": "35805",
    "Houston": "56003",
    "San Francisco": "60713",
    "Las Vegas": "45963",
    "Nashville": "55229",
    "Miami": "34438",
    "Atlanta": "60898",
    "Denver": "33388",
    "Louisville": "39604",
    "San Diego": "60750",
    "Dallas": "55711",
    "Seattle": "60878",
    "Phoenix": "31310",
    "Philadelphia": "60795",
    "San Antonio": "60956",
    "Honolulu": "60982",
    "New Orleans": "60864",
    "Orlando": "34515"
}

city_to_airport = { #maps city to airport code
    "New York City": "JFK",   # John F. Kennedy International Airport
    "Los Angeles": "LAX",     # Los Angeles International Airport
    "Chicago": "ORD",         # O'Hare International Airport
    "Houston": "IAH",         # George Bush Intercontinental Airport
    "San Francisco": "SFO",   # San Francisco International Airport
    "Las Vegas": "LAS",       # Harry Reid International Airport
    "Nashville": "BNA",       # Nashville International Airport
    "Miami": "MIA",           # Miami International Airport
    "Atlanta": "ATL",         # Hartsfield-Jackson Atlanta International Airport
    "Denver": "DEN",          # Denver International Airport
    "Louisville": "SDF",      # Louisville Muhammad Ali International Airport
    "San Diego": "SAN",       # San Diego International Airport
    "Dallas": "DFW",          # Dallas/Fort Worth International Airport
    "Seattle": "SEA",         # Seattle-Tacoma International Airport
    "Phoenix": "PHX",         # Phoenix Sky Harbor International Airport
    "Philadelphia": "PHL",    # Philadelphia International Airport
    "San Antonio": "SAT",     # San Antonio International Airport
    "Honolulu": "HNL",        # Daniel K. Inouye International Airport
    "New Orleans": "MSY",     # Louis Armstrong New Orleans International Airport
    "Orlando": "MCO"          # Orlando International Airport
}


class Message(Schema):
    sender = StringField(desc="The name of the messager")
    message = StringField(desc='The message to be received')
    
    application = StringField(desc='application used to send the send the message (ex. iMessage, gmail, instagram etc.)')

    attached_links = ListField(
        element_type=StringField,
        desc="A list of the attached links in the message",
    )

    attached_images = ListField(
        element_type=ImageFilepathField,
        desc="A list of the attached image file paths in the message",
    )

    other_receivers =ListField(
        element_type=StringField,
        desc="A list of other users who receive  the message",
    )

    time_passed = NumericField(desc = 'minutes passed since the message has come through')






class Importance(Schema):
    rating = NumericField(desc="A value quantitively expressing the importance of a message on a 0-100 scale")


class MessageDataReader(pz.DataReader):
    def __init__(self, message_dicts):
        super().__init__(Message)
        self.message_dicts = message_dicts
        # self.listings_dir = listings_dir
        # self.listings = sorted([dir for dir in os.listdir(self.listings_dir) if "listing" in dir])

    def __len__(self):
        return len(self.message_dicts)

    def __getitem__(self, idx: int):
        # fetch listing
        message = self.message_dicts[idx]
        output = {}
        for i in message:
            output[i] = message[i]
        
        return output






def is_imperative (message_ds):
    # print(active_ds)
    if message_ds is None or message_ds['rating'] is None:
        return False
    return message_ds['rating'] >= 70

if __name__ == "__main__":
    ##MY CODE BEGINS

    message1 = {
    "sender": "Alice Johnson",
    "message": "Hey team, just a reminder that our sync is tomorrow at 10 AM. I’ve attached the updated deck with slide notes and speaker assignments. Please make sure to review the transitions between your sections so it feels smooth. Let me know if you want to hop on a call to practice.",
    "application": "Outlook",
    "time_passed": "15 minutes",
    "other_receivers": ["Bob Smith", "Carla Liu"],
    "attached_links": ["https://companydocs.com/presentation"],
    "attached_images": ["slide_preview.png"]
    }

    message2 = {
    "sender": "Mike Chen",
    "message": "Check out this meme lol",
    "application": "iMessage",
    "time_passed": "2 minutes",
    "other_receivers": [],
    "attached_links": [],
    "attached_images": ["funny_cat.jpg"]
    }

    message3 = {
    "sender": "Sarah Patel",
    "message": "Final version of the Q2 report is attached. I’ve also included the breakdown of regional performance in case you want to pull any insights for your next strategy presentation.",
    "application": "Gmail",
    "time_passed": "30 minutes",
    "other_receivers": ["Jason Lee"],
    "attached_links": ["https://drive.google.com/file/d/1xyz123"],
    "attached_images": []
    }

    message4 = {
    "sender": "David Kim",
    "message": "Can we move the meeting to 3 PM?",
    "application": "Slack",
    "time_passed": "5 minutes",
    "other_receivers": ["Rachel Park"],
    "attached_links": [],
    "attached_images": []
    }

    message5 = {
    "sender": "Emily Zhou",
    "message": "We’re going with option 2 — it tested better with users and aligns with brand colors.",
    "application": "Teams",
    "time_passed": "8 minutes",
    "other_receivers": ["Liam Torres"],
    "attached_links": ["https://figma.com/project/design123"],
    "attached_images": []
    }

    message6 = {
    "sender": "Jake Thompson",
    "message": "The API docs are here. Let’s stick to v2 endpoints for now. Make sure to review auth headers — they’ve changed slightly in the latest update.",
    "application": "Gmail",
    "time_passed": "20 minutes",
    "other_receivers": ["Nina Gupta"],
    "attached_links": ["https://docs.api.com/v2"],
    "attached_images": []
    }

    message7 = {
    "sender": "Ava Lin",
    "message": "Hey! I booked my flights! Sharing the itinerary here. Can’t wait to see you in SF :)",
    "application": "Outlook",
    "time_passed": "12 minutes",
    "other_receivers": ["Tom Rivera"],
    "attached_links": ["https://delta.com/itinerary"],
    "attached_images": ["boarding_pass.png"]
    }

    message8 = {
    "sender": "Chris O’Neil",
    "message": "Re: Performance Review — I’ve gone through it and added comments inline. Let me know if you want to go over anything together.",
    "application": "Gmail",
    "time_passed": "25 minutes",
    "other_receivers": ["Manager HR"],
    "attached_links": [],
    "attached_images": []
    }

    message9 = {
    "sender": "Linda Gomez",
    "message": "Dinner tonight? The place I mentioned last time has an updated menu — link attached.",
    "application": "WhatsApp",
    "time_passed": "1 minute",
    "other_receivers": ["Sophia Rojas"],
    "attached_links": ["https://restaurant.com/menu"],
    "attached_images": []
    }

    message10 = {
    "sender": "Robert Wu",
    "message": "Attached is the invoice for March. Let me know if you need a separate copy for procurement.",
    "application": "Outlook",
    "time_passed": "10 minutes",
    "other_receivers": ["Accounts Payable"],
    "attached_links": [],
    "attached_images": ["invoice_march2025.pdf"]
    }

    message11 = {
    "sender": "Nora Davis",
    "message": "Updated mockups uploaded! We’ve added animations and improved button spacing for better UX.",
    "application": "Slack",
    "time_passed": "4 minutes",
    "other_receivers": ["UI Team"],
    "attached_links": ["https://drive.com/mockups"],
    "attached_images": ["mobile_ui_v2.png"]
    }

    message12 = {
    "sender": "Markus Evans",
    "message": "Can you please sign the NDA when you have a moment? We can’t proceed with the demo until it’s in.",
    "application": "Gmail",
    "time_passed": "35 minutes",
    "other_receivers": ["Legal Dept"],
    "attached_links": ["https://docusign.com/nda"],
    "attached_images": []
    }

    message13 = {
    "sender": "Julia Tran",
    "message": "Your code looks great! Just flagged one potential edge case in the data validation logic. Otherwise, solid work.",
    "application": "GitHub",
    "time_passed": "18 minutes",
    "other_receivers": [],
    "attached_links": ["https://github.com/org/repo/pull/42"],
    "attached_images": []
    }

    message14 = {
    "sender": "Leo Moreno",
    "message": "Yesterday’s hike was unreal! That last ridge view — wow. I’m dropping some pics here. We should totally do this again next month.",
    "application": "iMessage",
    "time_passed": "9 minutes",
    "other_receivers": ["Camila Diaz"],
    "attached_links": [],
    "attached_images": ["mountain_view.jpg", "sunset_peak.jpg"]
    }

    message15 = {
    "sender": "Tina Schwartz",
    "message": "Project timeline updated with the feedback from engineering and marketing. Note the shifted delivery date for Phase 2 — please flag if this is a blocker.",
    "application": "Teams",
    "time_passed": "22 minutes",
    "other_receivers": ["Engineering Lead"],
    "attached_links": ["https://notion.so/project-timeline"],
    "attached_images": []
    }

    message16 = {
    "sender": "Alex Yu",
    "message": "Hey, just dropping this here — RSVP by Friday if you’re coming to the launch party!",
    "application": "Gmail",
    "time_passed": "40 minutes",
    "other_receivers": ["Event Team"],
    "attached_links": ["https://eventbrite.com/x123"],
    "attached_images": []
    }

    message17 = {
    "sender": "Maya Ross",
    "message": "This podcast is a must-listen. It dives into how tech is reshaping the way we think — really makes you reflect.",
    "application": "iMessage",
    "time_passed": "3 minutes",
    "other_receivers": [],
    "attached_links": ["https://spotify.com/ep123"],
    "attached_images": []
    }

    message18 = {
    "sender": "Daniel Brooks",
    "message": "Hey team, receipt from our offsite lunch is attached. Also, huge thanks to everyone for making the event such a success!",
    "application": "Outlook",
    "time_passed": "14 minutes",
    "other_receivers": ["Office Manager"],
    "attached_links": [],
    "attached_images": ["receipt_april.jpg"]
    }

    message19 = {
    "sender": "Grace Lee",
    "message": "Attached are the onboarding slides for new interns. I’ve updated it to include the revised access instructions for GitHub, Jira, and VPN.",
    "application": "Gmail",
    "time_passed": "17 minutes",
    "other_receivers": ["HR Team"],
    "attached_links": [],
    "attached_images": ["orientation2025.pdf"]
    }

    message20 = {
    "sender": "Omar Hassan",
    "message": "New post up! This one’s about staying creative in the age of algorithms. Would love your thoughts in the comments.",
    "application": "Instagram",
    "time_passed": "6 minutes",
    "other_receivers": ["Editorial"],
    "attached_links": ["https://instagram.com/p/abc123"],
    "attached_images": ["post_thumb.jpg"]
}
    message21 = {
    "sender": "Omar Hassan",
    "message": "I'm in the hospital! Please come I need help!!",
    "application": "iMessage",
    "time_passed": "2 minutes",
    "other_receivers": [],
    "attached_links": [],
    "attached_images": []
}





    messages = [message1, message2, message3, message4, message5,
    message6, message7, message8, message9, message10,
    message11, message12, message13, message14, message15,
    message16, message17, message18, message19, message20, message21]
    
    ds = Dataset(MessageDataReader(messages)) # list of city names
    ds = ds.sem_add_columns(Importance)

    #do i need this line
    ds = ds.filter(is_imperative, depends_on="rating")

    config = QueryProcessorConfig(
        cache=False, #changed from nocache=True
        verbose=True,
        policy=MaxQuality(),
        execution_strategy="parallel",
        available_models=[Model.MIXTRAL, Model.GPT_4o_MINI, Model.GPT_4o_MINI_V]
    )
    data_record_collection = ds.run(config)
    print(data_record_collection.to_df())