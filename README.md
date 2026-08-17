At its core, LangGraph models agent workflows as graphs. You define the behavior of your agents using three key components:
State: A shared data structure that represents the current snapshot of your application. It can be any data type, but is typically defined using a shared state schema.
Nodes: Functions that encode the logic of your agents. They receive the current state as input, perform some computation or side-effect, and return an updated state.
Edges: Functions that determine which Node to execute next based on the current state. They can be conditional branches or fixed transitions.