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

### The cleanest modern example: DBSP

[DBSP](https://arxiv.org/abs/2203.16684) (Budiu, McSherry, Ryzhyk, Tannen; VLDB 2023
best paper) treats streams like digital signal processing does, with values in an
abelian group (for example, relations with signed multiplicities). It needs only four
operators:

| Operator | Definition | Meaning |
|---|---|---|
| Delay $z^{-1}$ | $z^{-1}(s)[t] = s[t-1]$, and $0$ at $t=0$ | Look at the previous value |
| Integrate $I$ | $I(s)[t] = \sum_{i \le t} s[i]$ | Changes → full state ("table") |
| Differentiate $D$ | $D(s)[t] = s[t] - s[t-1]$ | Full state → changes ("changelog") |
| Lift $\uparrow\!Q$ | $(\uparrow\!Q)(s)[t] = Q(s[t])$ | Apply an ordinary query at each time step |

Incremental computation, which is the core of streaming, becomes one equation:

$$
Q^{\Delta} = D \circ \uparrow\!Q \circ I
$$

From this it **proves** the rules that streaming engines use:

- **Chain rule:** $(Q_1 \circ Q_2)^{\Delta} = Q_1^{\Delta} \circ Q_2^{\Delta}$. So you can
  incrementalize each operator on its own.
- **Linear operators** (filter, project, union): $Q^{\Delta} = Q$. They are already
  incremental.
- **Join** is bilinear:
  $\Delta(A \bowtie B) = \Delta A \bowtie \Delta B + A_{t-1} \bowtie \Delta B + \Delta A \bowtie B_{t-1}$.
  This is exactly why a streaming join has to keep both sides as state.

The stream/table duality is also just the equation $I \circ D = \text{id}$.

### Other formal frameworks

| Framework | Formalism | Paper |
|---|---|---|
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

Read **DBSP** first. It's the most direct answer to "streaming as pure math", it is
short, and its four operators map straight onto how engines like Flink or Materialize
actually work. Read **Kahn 1974** if you want the foundational theory.


## Flink vs DBSP

They're related in idea but not in history. Flink wasn't built on DBSP; it's older and
grew out of the Stratosphere research project at TU Berlin, becoming Apache Flink
around 2014. But DBSP is a good **mathematical description of what Flink SQL already
does**. Flink's SQL/Table API reached the same design by engineering, and DBSP later
showed why that design works.

### How Flink SQL maps onto DBSP

| DBSP concept | Flink SQL equivalent |
|---|---|
| Stream of changes with signed weights (Z-sets: +1 insert, −1 delete) | **Changelog stream** with `RowKind`: `+I`, `-U`, `+U`, `-D` |
| Integrate $I$: changes → table | A changelog applied to state gives a [**dynamic table**](https://nightlies.apache.org/flink/flink-docs-stable/docs/dev/table/concepts/dynamic_tables/) |
| Differentiate $D$: table → changes | A dynamic table's query result gets emitted as a changelog |
| $I \circ D = \text{id}$ (stream/table duality) | Flink's "dynamic tables" concept: a stream *is* a table changing over time |
| Linear operators are already incremental | Filter and project pass changelog rows straight through, with no state |
| Join is bilinear: $\Delta A \bowtie \Delta B + A_{t-1} \bowtie \Delta B + \Delta A \bowtie B_{t-1}$ | A **regular join** keeps both inputs in state and joins each new row against the other side's history |
| Group inverses (subtraction) | Aggregate functions with `retract()`, so an update can take back its earlier contribution |

### Where they differ

| | DBSP | Flink |
|---|---|---|
| What it is | A theory, plus a reference implementation (Feldera, in Rust) | A production distributed engine |
| Incrementalization | **Derived automatically** for any query: $Q^{\Delta} = D \circ \uparrow\!Q \circ I$ | **Hand-written per operator**. The planner works out whether each operator's output is append-only or contains updates and retractions |
| Time | A logical clock: each step is one batch of changes | **Event time + watermarks**, with out-of-order data and late arrivals |
| Scope | Relational queries, including recursion | SQL *and* the DataStream API: arbitrary user code, not limited to group-structured data |
| Fault tolerance | Not part of the theory | Asynchronous barrier snapshots, a Chandy–Lamport variant ([Carbone et al., 2015](https://arxiv.org/abs/1506.08603)) |

So DBSP explains Flink **SQL** well, but not Flink as a whole. The DataStream API,
watermarks and checkpointing are operational parts that sit outside DBSP's equations.

### The family tree

```
Stratosphere (TU Berlin) ──► Apache Flink ──── changelogs / dynamic tables (engineering)
Naiad / differential dataflow (McSherry) ──► Materialize
                         └──► DBSP (Budiu, McSherry, et al.) ──► Feldera
```

The engines that **actually use** these models are Materialize (differential dataflow)
and Feldera (DBSP). Flink arrived at a similar design separately.

### Why it matters in practice

The cases where Flink SQL is hard to work with, like retraction storms, joins whose
state grows without limit, or non-deterministic updates, are mostly the places where
Flink's hand-written incremental rules depart from what the DBSP algebra would derive.
The DBSP equations make it clear which state an operator *must* keep. For example, the
join formula needs $A_{t-1}$ and $B_{t-1}$, the full history of both sides, unless a
window or a TTL bounds them.


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
* Carbone, Fóra, Ewen, Haridi, Tzoumas. *Lightweight Asynchronous Snapshots for Distributed Dataflows*. 2015. [arXiv:1506.08603](https://arxiv.org/abs/1506.08603)
* Apache Flink documentation. [*Dynamic Tables*](https://nightlies.apache.org/flink/flink-docs-stable/docs/dev/table/concepts/dynamic_tables/).
