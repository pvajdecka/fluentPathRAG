from __future__ import annotations
from dataclasses import dataclass
from typing import List

@dataclass
class Node:
    id: str
    text: str

@dataclass
class Edge:
    src: str
    dst: str
    relation: str

@dataclass
class Path:
    nodes: List[Node]
    edges: List[Edge]  # edges[i] connects nodes[i] -> nodes[i+1]

@dataclass
class Example:
    query: str
    paths: List[Path]
    gold_refs: List[str]


def build_examples() -> List[Example]:
    """Five multi-hop examples with long-form, single-sentence answers."""
    exs: List[Example] = []

    # 1) Characteristics of Mercury
    ME0 = Node("ME0", "Mercury is the closest planet to the Sun.")
    ME1 = Node("ME1", "Mercury has a near‑vacuum exosphere and virtually no atmosphere.")
    ME2 = Node("ME2", "Mercury experiences extreme day–night temperature variation.")
    ME3 = Node("ME3", "Mercury rotates slowly in a 3:2 spin–orbit resonance.")
    ME4 = Node("ME4", "Mercury’s surface is heavily cratered.")
    p1a = Path(nodes=[ME0, ME1], edges=[Edge("ME0", "ME1", "has_property")])
    p1b = Path(nodes=[ME0, ME2], edges=[Edge("ME0", "ME2", "has_property")])
    p1c = Path(nodes=[ME0, ME3], edges=[Edge("ME0", "ME3", "has_spin_orbit")])
    p1d = Path(nodes=[ME0, ME4], edges=[Edge("ME0", "ME4", "has_surface_feature")])
    exs.append(Example(
        query="What are defining characteristics of Mercury?",
        paths=[p1a, p1b, p1c, p1d],
        gold_refs=[
            "Mercury, the innermost planet, has a near‑vacuum exosphere with virtually no atmosphere, extreme day–night temperature swings, a 3:2 spin–orbit resonance with slow rotation, and a heavily cratered surface.",
            "Key traits of Mercury include its position closest to the Sun, a tenuous exosphere, drastic thermal variation, slow 3:2 spin–orbit rotation, and a battered, cratered terrain.",
            "As the closest planet, Mercury exhibits a thin exosphere instead of a true atmosphere, large diurnal temperature differences, slow 3:2 rotational locking, and a surface marked by extensive cratering."
        ]
    ))

    # 2) Why colonization began
    C1 = Node("C1", "Christopher Columbus sailed west from Spain in 1492 seeking a route to Asia.")
    C2 = Node("C2", "Columbus landed in the Caribbean, initiating sustained European contact with the Americas.")
    C3 = Node("C3", "European powers sought trade routes, gold and spices, and territorial expansion.")
    C4 = Node("C4", "The Spanish monarchy financed Columbus’s voyage.")
    p2a = Path(nodes=[C4, C1, C2, C3], edges=[
        Edge("C4", "C1", "financed"),
        Edge("C1", "C2", "voyage_landed_in"),
        Edge("C2", "C3", "led_to")
    ])
    exs.append(Example(
        query="Why did European colonization in the Americas begin?",
        paths=[p2a],
        gold_refs=[
            "European colonization of the Americas began after Columbus’s 1492 voyage from Spain reached the Caribbean, catalyzing sustained contact driven by monarchic sponsorship, trade ambitions, gold and spices, and expanding imperial influence.",
            "Columbus’s Caribbean landfall in 1492, financed by the Spanish crown, opened enduring European contact that was motivated by new trade routes, precious resources, and territorial expansion.",
            "Backed by Spain, Columbus’s voyage connected Europe to the Americas, and the pursuit of Asian commerce, wealth, and power subsequently spurred large‑scale colonization."
        ]
    ))

    # 3) Smoking → DNA damage → malignancy → mortality
    S1 = Node("S1", "Tobacco smoke contains carcinogens.")
    S2 = Node("S2", "Carcinogens cause DNA damage and mutations in lung cells.")
    S3 = Node("S3", "Accumulated mutations can drive malignant transformation and tumor growth.")
    S4 = Node("S4", "Lung cancer is a leading cause of cancer death.")
    S5 = Node("S5", "Smoking substantially increases the risk of lung cancer.")
    p3a = Path(nodes=[S1, S2, S3, S4], edges=[
        Edge("S1", "S2", "exposure_causes"),
        Edge("S2", "S3", "leads_to"),
        Edge("S3", "S4", "results_in")
    ])
    p3b = Path(nodes=[S5], edges=[])
    exs.append(Example(
        query="How does smoking lead to lung cancer death?",
        paths=[p3a, p3b],
        gold_refs=[
            "Smoking delivers carcinogens that damage DNA in lung cells, and as mutations accumulate they enable malignant transformation and tumor growth, making lung cancer a leading cause of cancer death.",
            "Tobacco smoke’s carcinogens induce DNA lesions that, over time, create oncogenic mutations, drive uncontrolled proliferation, and ultimately cause lethal lung cancer in many smokers.",
            "By exposing lung tissue to carcinogens, smoking causes mutational damage that leads to malignant change and aggressive tumors, explaining lung cancer’s prominence among cancer deaths."
        ]
    ))

    # 4) Neural networks training
    N1 = Node("N1", "A neural network performs a forward pass that computes activations and predictions.")
    N2 = Node("N2", "A loss function measures the error between predictions and targets.")
    N3 = Node("N3", "Backpropagation computes gradients of the loss with respect to the weights.")
    N4 = Node("N4", "Gradient descent or its variants update weights using those gradients.")
    N5 = Node("N5", "Training iterates over minibatches and epochs, often with regularization to improve generalization.")
    p4a = Path(nodes=[N1, N2, N3, N4], edges=[
        Edge("N1", "N2", "loss_computed_from"),
        Edge("N2", "N3", "backprop_computes_gradients"),
        Edge("N3", "N4", "optimizer_updates")
    ])
    p4b = Path(nodes=[N4, N5], edges=[Edge("N4", "N5", "repeats_over_data")])
    exs.append(Example(
        query="How are neural networks trained?",
        paths=[p4a, p4b],
        gold_refs=[
            "Neural networks are trained by forward passes that produce predictions, loss functions measuring error, backpropagation computing gradients, and gradient‑descent optimizers updating weights over many minibatches and epochs with possible regularization.",
            "Training proceeds by computing outputs, evaluating loss against targets, propagating gradients backward, and adjusting parameters via gradient descent repeatedly across batches and epochs, often with techniques like regularization or momentum.",
            "Learning involves forward computation, loss evaluation, gradient backpropagation, and iterative optimizer updates across the dataset, typically for multiple epochs with optional regularization to improve generalization."
        ]
    ))

    # 5) Beethoven chain
    BE1 = Node("BE1", "Ludwig van Beethoven was born in Bonn.")
    BE2 = Node("BE2", "Bonn is a city in Germany.")
    BE3 = Node("BE3", "Berlin is the capital of Germany.")
    BE4 = Node("BE4", "The Spree River flows through Berlin.")
    p5a = Path(nodes=[BE1, BE2, BE3], edges=[
        Edge("BE1", "BE2", "born_in_city"),
        Edge("BE2", "BE3", "country_has_capital")
    ])
    p5b = Path(nodes=[BE3, BE4], edges=[Edge("BE3", "BE4", "capital_has_river")])
    exs.append(Example(
        query=(
            "Which river flows through the capital of the country where Beethoven was born, "
            "and how does that relationship follow from the relevant locations?"
        ),
        paths=[p5a, p5b],
        gold_refs=[
            "Because Beethoven was born in Bonn in Germany, whose capital is Berlin through which the Spree River flows, the river connected to the capital of his birth country is the Spree.",
            "Beethoven’s birthplace, Bonn, lies in Germany, and Germany’s capital is Berlin, a city traversed by the Spree River, making the relevant river the Spree.",
            "Bonn, Beethoven’s birthplace, is in Germany; Germany’s capital is Berlin, and Berlin is crossed by the Spree River, so the river in question is the Spree."
        ]
    ))

    return exs