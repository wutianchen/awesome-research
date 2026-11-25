# DBSP

A mathematical model of incremental stream processing: a whole streaming pipeline
written as equations over streams. For how it compares with other formal models of
streaming, see [Stream processing](stream-processing.md).

## The model

**DBSP** ([Budiu, McSherry, Ryzhyk, Tannen](https://arxiv.org/abs/2203.16684); VLDB 2023
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

### Where it is used

[Feldera](https://github.com/feldera/feldera) is the engine built on DBSP (in Rust) by
the paper's authors. Its SQL compiles to DBSP circuits, so every query is incremental by
construction.


## DBSP vs Flink

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
* McSherry, Murray, Isaacs, Isard. *Differential Dataflow*. CIDR 2013.
* Carbone, Fóra, Ewen, Haridi, Tzoumas. *Lightweight Asynchronous Snapshots for Distributed Dataflows*. 2015. [arXiv:1506.08603](https://arxiv.org/abs/1506.08603)
* Apache Flink documentation. [*Dynamic Tables*](https://nightlies.apache.org/flink/flink-docs-stable/docs/dev/table/concepts/dynamic_tables/).
* [Feldera](https://github.com/feldera/feldera) on GitHub.
