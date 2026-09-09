#!/usr/bin/env python3
"""One functional taxonomy of agent roles, shared by every figure that draws them.

The concrete role names differ between the figures because they come from different sources: the
case study names its own agents, the orchestration architectures use the names of the study that
defined them, and the infrastructure benchmark uses the keys of its own configuration files. Those
names are left alone -- renaming them would break traceability to the artefacts -- and the figures
are unified by functional CLASS instead, carried by colour.

    generate    produces a candidate or a plan       planner, proposals (L), proposer
    transform   refines ONE line of work             analyst, writer, critic, refiner, verifier
    ground      acquires state external to W        retriever, inspector
    aggregate   reduces COMPETING candidates        manager, synthesiser, duelist

The transform/aggregate line is drawn at competing candidates, not at fan-in: a cascading refiner
reads its own history and still refines one draft, whereas a duelist that picks one of two rival
answers, and a synthesiser that merges three, both reduce a candidate set.

GROUND has no counterpart in the orchestration taxonomy, which has neither tools nor retrieval.
That absence is a result rather than an oversight: it is the class this paper adds.
"""

GENERATE  = "#4C78A8"
TRANSFORM = "#F58518"
GROUND    = "#54A24B"
AGGREGATE = "#333333"

LABEL = {GENERATE:  "generate",
         TRANSFORM: "transform",
         GROUND:    "ground",
         AGGREGATE: "aggregate"}

# concrete role name -> class, across all three figures
ROLE_CLASS = {
    "planner": GENERATE, "proposer": GENERATE, "proposal": GENERATE,
    "analyst": TRANSFORM, "writer": TRANSFORM, "critic": TRANSFORM,
    "refiner": TRANSFORM, "verifier": TRANSFORM,
    "retriever": GROUND, "inspector": GROUND, "inspection": GROUND,
    "manager": AGGREGATE, "synthesiser": AGGREGATE, "synthesizer": AGGREGATE,
    "duelist": AGGREGATE,
}


def role_class(name):
    """Class colour for a concrete role name; KeyError is deliberate, so new roles are classified."""
    return ROLE_CLASS[name.strip().lower()]


def legend_entries(classes):
    """(colour, label) pairs, in a fixed order, for the classes a figure actually uses."""
    order = (GENERATE, TRANSFORM, GROUND, AGGREGATE)
    return [(c, LABEL[c]) for c in order if c in classes]
