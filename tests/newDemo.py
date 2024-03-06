#!/usr/bin/env python3
from palimpzest.elements import Schema, StringField
from palimpzest.policy import *

import palimpzest as pz

import json
import time


def emitNestedTuple(node, indent=0):
    elt, child = node
    print(" " * indent, elt)
    if child is not None:
        emitNestedTuple(child, indent=indent+2)


# TODO: I want this to "just work" if it inherits from Schema instead of TextFile;
#       for some reason, inheriting from Schema leads to the "contents" being a bytes
#       field but if Email inherits from File or TextFile, it becomes a string;
#       this is important b/c dr.asTextJSON() will ignore bytes field(s).
class Email(pz.TextFile):
    """Represents an email, which in practice is usually from a text file"""
    sender = StringField(desc="The email address of the sender", required=True)
    subject = StringField(desc="The subject of the email", required=True)


if __name__ == "__main__":
    """
    This demo illustrates how the cost optimizer can produce and evaluate multiple plans.
    """
    startTime = time.time()

    # user implemented plan
    emails = pz.Dataset(source="enron-tiny", schema=Email)
    emails = emails.filterByStr("The email is about someone taking a vacation")
    emails = emails.filterByStr("The email is sent by Larry")

    # get logical tree
    t1 = time.time()
    logicalTree = emails.getLogicalTree()
    t2 = time.time()

    # get candidate physical plans
    candidatePlans = logicalTree.createPhysicalPlanCandidates()
    t3 = time.time()
    print(f"Create Plan: {t1 - startTime:.3f}")
    print(f"Get Logical Tree: {t2 - t1:.3f}")
    print(f"Create Cand. Plans: {t3 - t2:.3f}")

    # use sampling to get better information about plans
    # sampler = SimpleSampler(min=10)
    # candidatePlans = logicalTree.createPhysicalPlanCandidates(sampler=sampler)

    # print out plans to the user
    print("----------")
    for idx, cp in enumerate(candidatePlans):
        print(f"Plan {idx}: Time est: {cp[0]:.3f} -- Cost est: {cp[1]:.3f} -- Quality est: {cp[2]:.3f}")
        print("Physical operator tree")
        physicalOps = cp[3].dumpPhysicalTree()
        emitNestedTuple(physicalOps)
        print("----------")

    # have policy select the candidate plan to execute
    myPolicy = UserChoice()
    planTime, planCost, quality, physicalTree = myPolicy.choose(candidatePlans)
    print("----------")
    print(f"Policy is: {str(myPolicy)}")
    print(f"Chose plan: Time est: {planTime:.3f} -- Cost est: {planCost:.3f} -- Quality est: {quality:.3f}")
    emitNestedTuple(physicalTree.dumpPhysicalTree())
    t4 = time.time()

    # execute the plan
    all_timing_info = []
    for timing_info, r in physicalTree:
        print(f"(sender={r.sender}, subject={r.subject})")
        all_timing_info.append(timing_info)

    with open('timing_info.json', 'w') as f:
        json.dump(all_timing_info, f)

    endTime = time.time()
    print(f"Plan Selection Time: {t4 - t3}")
    print(f"Execution Time: {endTime - t4}")
    print(f"Total Elapsed Time: {endTime - startTime}")
