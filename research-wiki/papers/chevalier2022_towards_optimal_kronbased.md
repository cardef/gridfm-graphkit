---
type: paper
node_id: paper:chevalier2022_towards_optimal_kronbased
title: "Towards Optimal Kron-based Reduction Of Networks (Opti-KRON) for the Electric Power Grid"
authors: ["Samuel Chevalier", "Mads R. Almassalkhi"]
year: 2022
venue: "arXiv"
external_ids:
  arxiv: "2204.05554"
  doi: null
  s2: null
tags: ["kron", "network-reduction"]
added: 2026-07-07T17:20:57Z
---

# Towards Optimal Kron-based Reduction Of Networks (Opti-KRON) for the Electric Power Grid

## One-line thesis
Opti-KRON: MILP-optimal Kron-based reduction trading reduction depth against voltage-error on an AC load-flow data library (25-85% reduction, <0.01 pu error).

## Problem / Gap
_TODO._

## Method
_TODO._

## Key Results
_TODO._

## Assumptions
_TODO._

## Limitations / Failure Modes
_TODO._

## Reusable Ingredients
_TODO._

## Open Questions
_TODO._

## Claims
_TODO._

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._

## Relevance to This Project
_TODO._

## Abstract (original)

> For fast timescales or long prediction horizons, the AC optimal power flow (OPF) problem becomes a computational challenge for large-scale, realistic AC networks. To overcome this challenge, this paper presents a novel network reduction methodology that leverages an efficient mixed-integer linear programming (MILP) formulation of a Kron-based reduction that is optimal in the sense that it balances the degree of the reduction with resulting modeling errors in the reduced network. The method takes as inputs the full AC network and a pre-computed library of AC load flow data and uses the graph Laplacian to constraint nodal reductions to only be feasible for neighbors of non-reduced nodes. This results in a highly effective MILP formulation which is embedded within an iterative scheme to successively improve the Kron-based network reduction until convergence. The resulting optimal network reduction is, thus, grounded in the physics of the full network. The accuracy of the network reduction methodology is then explored for a 100+ node medium-voltage radial distribution feeder example across a wide range of operating conditions. It is finally shown that a network reduction of 25-85% can be achieved within seconds and with worst-case voltage magnitude deviation errors within any super node cluster of less than 0.01pu. These results illustrate that the proposed optimization-based approach to Kron reduction of networks is viable for larger networks and suitable for use within various power system applications.
