# Stream processing

## Can stream processing be expressed purely in math?

Yes. Several formal frameworks describe stream processing entirely as equations over
streams. The usual split is that equations capture **what** a pipeline computes, while
**how** it runs (latency, out-of-order arrival, fault tolerance) needs extra modelling on
top.

### The core idea

Model a stream as a **function of time**, and an operator as a **function from streams
to streams**:

$$
s : \mathbb{N} \to A \qquad\qquad Q : (\mathbb{N} \to A) \to (\mathbb{N} \to B)
$$

A pipeline is then a composition of such functions. Feedback loops become equations that
the stream must satisfy, solved as a fixed point.

### Formal frameworks

| Framework | Formalism | Paper |
|---|---|---|
| **[DBSP](DBSP.md)** | Streams as signals over an abelian group, with four operators (delay, integrate, differentiate, lift). Incremental computation is one equation, $Q^{\Delta} = D \circ \uparrow\!Q \circ I$ | Budiu, McSherry, Ryzhyk, Tannen, *DBSP*, VLDB 2023 |
| **Kahn process networks** | A process is a continuous function on streams ordered by prefix. A network is the **least fixed point** of its equations. This is the classic denotational semantics of dataflow | Kahn, *The Semantics of a Simple Language for Parallel Programming*, IFIP 1974 |
| **Lucid / Lustre** (synchronous dataflow) | Programs are sets of equations over infinite sequences, such as `x = 0 fby (x + 1)`. Lustre is used in avionics (SCADE) | Halbwachs, Caspi et al., *The Synchronous Data Flow Programming Language LUSTRE*, Proc. IEEE 1991 |
| **Stream calculus** (coinduction) | Streams as a final coalgebra. Operators are defined by *stream differential equations*, giving an algebra much like power series | Rutten, *A Coinductive Calculus of Streams*, MSCS 2005 |
| **FOCUS** | Systems specified as relations between input and output stream histories, with a theory of refinement | Broy & Stølen, *Specification and Development of Interactive Systems*, Springer 2001 |
| **CQL** | Three operator classes: stream→relation (windows), relation→relation (SQL), relation→stream (Istream/Dstream/Rstream) | Arasu, Babu, Widom, *The CQL Continuous Query Language*, VLDB Journal 2006 |
| **Differential dataflow** | Collections as functions of a **partially ordered** time (for example, epoch × loop iteration) with difference traces, which handles iteration incrementally | McSherry et al., *Differential Dataflow*, CIDR 2013 |
| **Time-varying relations** | Streaming SQL as relations indexed by time, with tables and streams as two views of the same thing | Begoli et al., *One SQL to Rule Them All*, SIGMOD 2019 |
| **Monoidal streams** (category theory) | Dataflow programs with feedback as morphisms in a monoidal category | Di Lavore, de Felice, Román, *Monoidal Streams for Dataflow Programming*, LICS 2022 |
| **Dataflow model** (less formal) | Formalizes event time vs processing time, windows, watermarks and triggers as separate axes: *what*, *where*, *when*, *how* | Akidau et al., *The Dataflow Model*, VLDB 2015 |

### What the equations don't capture easily

| Aspect | Formal status |
|---|---|
| The result of a query over a stream | Fully formal (DBSP, Kahn, CQL) |
| Windows, event time | Formal: an index on time plus a function that assigns records to windows |
| **Out-of-order data, watermarks, late data** | Partly formal. The Dataflow model and differential dataflow's partial-order time come closest |
| Latency, throughput, state size | Operational. They need cost models or queueing theory, not the semantics |
| Fault tolerance, exactly-once | Separate formalisms: distributed snapshots (Chandy–Lamport 1985) and the protocol proofs built on them |

### Where to start

Read **[DBSP](DBSP.md)** first. It's the most direct answer to "streaming as pure math", it is
short, and its four operators map straight onto how engines like Flink or Materialize
actually work (see [DBSP vs Flink](DBSP.md#dbsp-vs-flink)). Read **Kahn 1974** if you want the
foundational theory.


# Reference

* Budiu, McSherry, Ryzhyk, Tannen. *DBSP: Automatic Incremental View Maintenance for Rich Query Languages*. VLDB 2023. [arXiv:2203.16684](https://arxiv.org/abs/2203.16684)
* Kahn. *The Semantics of a Simple Language for Parallel Programming*. IFIP Congress 1974.
* Halbwachs, Caspi, Raymond, Pilaud. *The Synchronous Data Flow Programming Language LUSTRE*. Proceedings of the IEEE, 1991.
* Rutten. *A Coinductive Calculus of Streams*. Mathematical Structures in Computer Science, 2005.
* Broy, Stølen. *Specification and Development of Interactive Systems: Focus on Streams, Interfaces, and Refinement*. Springer, 2001.
* Arasu, Babu, Widom. *The CQL Continuous Query Language: Semantic Foundations and Query Execution*. VLDB Journal, 2006.
* McSherry, Murray, Isaacs, Isard. *Differential Dataflow*. CIDR 2013.
* Begoli, Akidau, Hueske, Hyde, Knight, Knowles. *One SQL to Rule Them All*. SIGMOD 2019.
* Di Lavore, de Felice, Román. *Monoidal Streams for Dataflow Programming*. LICS 2022.
* Akidau et al. *The Dataflow Model*. VLDB 2015.
* Chandy, Lamport. *Distributed Snapshots: Determining Global States of Distributed Systems*. ACM TOCS, 1985.
