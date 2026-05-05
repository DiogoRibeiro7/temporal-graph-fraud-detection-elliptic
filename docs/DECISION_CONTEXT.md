# Decision Context

This project frames graph fraud detection as investigation support.

## Decision being supported

Which transactions should analysts review first?

## Why graph structure matters

Suspicious activity is often relational. A transaction may be more informative when seen in the context of its neighbours, flows, and graph components.

## Why temporal validation matters

Fraud models are used forward in time. Random splits may leak future structure and labels into training, making the model look better than it is.

## Safe-use principle

Risk scores should support review. They should not become automatic enforcement decisions.
