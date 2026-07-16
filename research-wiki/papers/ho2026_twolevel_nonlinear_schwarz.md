---
type: paper
node_id: paper:ho2026_twolevel_nonlinear_schwarz
title: "Two-level nonlinear Schwarz methods - a parallel implementation with application to nonlinear elasticity and incompressible flow problems"
authors: ["Kyrill Ho", "Axel Klawonn", "Martin Lanser"]
year: 2026
venue: "arXiv"
external_ids:
  arxiv: "2603.24542"
  doi: null
  s2: null
tags: ["nonlinear-schwarz", "domain-decomposition"]
added: 2026-07-07T17:20:58Z
---

# Two-level nonlinear Schwarz methods - a parallel implementation with application to nonlinear elasticity and incompressible flow problems

## One-line thesis
Two-level nonlinear Schwarz (ASPIN family) with parallel implementation - nonlinear elimination of subdomain unknowns, i.e. a nonlinear Schur complement with coarse level.

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

> Nonlinear Schwarz methods are a type of nonlinear domain decomposition method used as an alternative to Newton's method for solving discretized nonlinear partial differential equations. In this article, the first parallel implementation of a two-level nonlinear Schwarz method leveraging the GDSW-type coarse spaces from the Fast and Robust Overlapping Schwarz (FROSch) framework in Trilinos is presented. This framework supports both additive and hybrid two-level nonlinear Schwarz methods and makes use of modifications to the coarse spaces constructed by FROSch to further enhance the robustness and convergence speed of the methods. Efficiency and excellent parallel performance of the software framework are demonstrated by applying it to two challenging nonlinear problems: the two-dimensional lid-driven cavity problem at high Reynolds numbers, and a Neo-Hookean beam deformation problem. The results show that two-level nonlinear Schwarz methods scale exceptionally well up to 9\,000 subdomains and are more robust than standard Newton-Krylov-Schwarz solvers for the considered Navier-Stokes problems with high Reynolds numbers or, respectively, for the nonlinear elasticity problems and large deformations. The new parallel implementation provides a foundation for future research in scalable nonlinear domain decomposition methods and demonstrates the practical viability of nonlinear Schwarz techniques for large-scale simulations.
