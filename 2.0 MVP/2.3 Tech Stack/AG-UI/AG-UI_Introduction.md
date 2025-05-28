Get Started
Introduction
Get started with the Agent User Interaction Protocol (AG-UI)

AG-UI standardizes how front-end applications connect to AI agents through an open protocol. Think of it as a universal translator for AI-driven systems- no matter what language an agent speaks: AG-UI ensures fluent communication.

​
Why AG-UI?
AG-UI helps developers build next-generation AI workflows that need real-time interactivity, live state streaming and human-in-the-loop collaboration.

AG-UI provides:

A straightforward approach to integrating AI agents with the front-end through frameworks such as CopilotKit 🪁
Building blocks for an efficient wire protocol for human⚡️agent communication
Best practices for chat, streaming state updates, human-in-the-loop and shared state
​
Existing Integrations
AG-UI has been integrated with several popular agent frameworks, making it easy to adopt regardless of your preferred tooling:

LangGraph: Build agent-native applications with shared state and human-in-the-loop workflows using LangGraph’s powerful orchestration capabilities.
CrewAI Flows: Create sequential multi-agent workflows with well-defined stages and process control.
CrewAI Crews: Design collaborative agent teams with specialized roles and inter-agent communication.
Mastra: Leverage TypeScript for building strongly-typed agent implementations with enhanced developer experience.
AG2: Utilize the open-source AgentOS for scalable, production-ready agent deployments.
These integrations make it straightforward to connect your preferred agent framework with frontend applications through the AG-UI protocol.

​
Architecture
At its core, AG-UI bridges AI agents and front-end applications using a lightweight, event-driven protocol:

Backend

Frontend

AG-UI Protocol

AG-UI Protocol

AG-UI Protocol

AG-UI Protocol

Front-end

AI Agent A

Secure Proxy

AI Agent B

AI Agent C

Front-end: The application (chat or any AI-enabled app) that communicates over AG-UI
AI Agent A: An agent that the front-end can connect to directly without going through the proxy
Secure Proxy: An intermediary proxy that securely routes requests from the front-end to multiple AI agents
Agents B and C: Agents managed by the proxy service
​
Technical Overview
AG-UI is designed to be lightweight and minimally opinionated, making it easy to integrate with a wide range of agent implementations. The protocol’s flexibility comes from its simple requirements:

Event-Driven Communication: Agents need to emit any of the 16 standardized event types during execution, creating a stream of updates that clients can process.

Bidirectional Interaction: Agents accept input from users, enabling collaborative workflows where humans and AI work together seamlessly.

The protocol includes a built-in middleware layer that maximizes compatibility in two key ways:

Flexible Event Structure: Events don’t need to match AG-UI’s format exactly—they just need to be AG-UI-compatible. This allows existing agent frameworks to adapt their native event formats with minimal effort.

Transport Agnostic: AG-UI doesn’t mandate how events are delivered, supporting various transport mechanisms including Server-Sent Events (SSE), webhooks, WebSockets, and more. This flexibility lets developers choose the transport that best fits their architecture.

This pragmatic approach makes AG-UI easy to adopt without requiring major changes to existing agent implementations or frontend applications.

​
Comparison with other protocols
AG-UI focuses explicitly and specifically on the agent-user interactivity layer. It does not compete with protocols such as A2A (Agent-to-Agent protocol) and MCP (Model Context Protocol).

For example, the same agent may communicate with another agent via A2A while communicating with the user via AG-UI, and while calling tools provided by an MCP server.

These protocols serve complementary purposes in the agent ecosystem:

AG-UI: Handles human-in-the-loop interaction and streaming UI updates
A2A: Facilitates agent-to-agent communication and collaboration
MCP: Standardizes tool calls and context handling across different models
​
Quick Start
Choose the path that fits your needs:

Build with AG-UI
Implement AG-UI events directly in your agent framework

Connect to AG-UI
Connect AG-UI with existing protocols or custom solutions

​
Resources
Explore guides, tools, and integrations to help you build, optimize, and extend your AG-UI implementation. These resources cover everything from practical development workflows to debugging techniques.

Explore Integrations
Discover ready-to-use AG-UI integrations across popular agent frameworks and platforms

Developing with Cursor
Use Cursor to build AG-UI implementations faster

Troubleshooting AG-UI
Fix common issues when working with AG-UI servers and clients

​
Explore AG-UI
Dive deeper into AG-UI’s core concepts and capabilities:

Core architecture
Understand how AG-UI connects agents, protocols, and front-ends

Transports
Learn about AG-UI’s communication mechanism

​
Contributing
Want to contribute? Check out our Contributing Guide to learn how you can help improve AG-UI.

​
Support and Feedback
Here’s how to get help or provide feedback:

For bug reports and feature requests related to the AG-UI specification, SDKs, or documentation (open source), please create a GitHub issue
For discussions or Q&A about the AG-UI specification, use the specification discussions
For discussions or Q&A about other AG-UI open source components, use the organization discussions
Build with AG-UI
github
