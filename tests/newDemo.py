#!/usr/bin/env python3
from palimpzest.elements import Filter, Schema, StringField

from typing import List

import palimpzest as pz

import time


class Email(Schema):
    """Represents an email, which in practice is usually from a text file"""
    sender = StringField(desc="The email address of the sender", required=True)
    subject = StringField(desc="The subject of the email", required=True)

# pz.Plan base-class has an optional plan() function which may or may not be overridden
# - if plan() is overridden, we take the logical plan implied by the implementation and search for best physical plan
# - otherwise, we enumerate multiple logical plans, compute physical plans for those logical plans, and propose or select best one(s)
class LarryVacationEmailsImperative(pz.Plan):
    """
    An example of how users can imperatively describe the set of emails they want
    from the `enron-tiny` dataset. In this example, the source is defined by its
    local filepath. The PZ planner will create a physical plan to process this dataset.
    """
    def plan(self) -> pz.Dataset:
        # NOTE: source path assumes this script is run from root of PZ repo
        emails = pz.Dataset(
            dataset_id="enron-tiny-emails",
            source="testdata/enron-tiny/",
            schema=Email,
        )
        emails = emails.filterByStr("The email is about someone taking a vaction")
        emails = emails.filterByStr("The email is sent by Larry")

        return emails

# TODO: not fully implemented yet
class LarryVacationEmailsDeclarative(pz.Plan):
    """
    Provide declarative statement of emails you want from `enron-tiny` dataset,
    which is pre-registered with PZ in this example. The PZ planner will jointly
    search for the best logical and physical plans to process this dataset.
    This enables us to do things like filter re-ordering at the logical level
    to further optimize data extraction.
    """
    def sources(self) -> List[pz.Dataset]:
        emails = [pz.Dataset(dataset_id="enron-tiny-emails", source="enron-tiny", schema=Email)]

        return emails

    def filters(self) -> List[Filter]:
        vacation_filter = Filter("The email is about someone taking a vaction", dataset_id="enron-tiny-emails")
        sender_filter = Filter("The email is sent by Larry", dataset_id="enron-tiny-emails")
        filters = [vacation_filter, sender_filter]

        return filters


if __name__ == "__main__":
    """
    This demo illustrates how to execute an imperative and declarative
    data extraction program using Palimpzest.
    """
    startTime = time.time()

    # imperative plan; user defines logical operations and pz.Plan only finds best physical counterpart
    imperativeEmailsPlan = LarryVacationEmailsImperative()
    imperativeEmailsPlan.execute()

    # declarative plan; user defines set of data they want, pz.Plan uses tricks like filter reordering
    # to pick best logical plan, while also considering the best associated physical plan(s)
    declarativeEmailsPlan = LarryVacationEmailsDeclarative()
    declarativeEmailsPlan.execute()

    endTime = time.time()
    print("Elapsed time:", endTime - startTime)
