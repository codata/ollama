# ODRL Support in Ollama

This document describes the integration of **ODRL (Open Digital Rights Language)** and **Decentralized Identifiers (DIDs)** within the Ollama ecosystem. These features enable data sovereignty, verifiable provenance, and explicit attribution for model outputs.

## 🛡️ Feature Overview

Ollama now supports identifying instances via **DIDs**. By linking an Ollama instance to a DID, every generation, chat response, and model listing can be cryptographically attributed to its owner. This provides a foundation for enforcing usage policies (ODRL Agreements) in distributed AI environments.

### Key Capabilities:
- **Server Identity**: The server can discover its identity from a local ODRL wallet (`~/.odrl/did.json`).
- **Attribution**: Generation and Chat responses include the server's DID for transparency.
- **Discovery**: Clients can query the server's identity via the `/api/did` endpoint.
- **Model Metadata**: Listings and process statuses now include the identity of the serving instance.

## 🛠️ Implementation Details

### 1. The `odrl` Package
A new internal package `github.com/ollama/ollama/odrl` handles the discovery and loading of the instance identity.

- **Storage**: Identity is read from `~/.odrl/did.json`.
- **Primary Method**: `odrl.GetDID()` returns the master DID as a string (e.g., `did:oyd:zQmc...`).

### 2. API Extensions
New fields have been added to the following API response structs:
- **Generate**: Includes `did` in the JSON response.
- **Chat**: Includes `did` in the JSON response.
- **Tags & List**: Includes `did` for every model available on the instance.
- **Process (ps)**: Shows which instance identity is currently running a model.
- **Show**: Provides the serving identity for model details.

### 3. New Endpoint: `/api/did`
Returns the server's DID in a standard JSON format.
```bash
curl http://localhost:11434/api/did
# Output: {"did": "did:oyd:zQmcVHWDMe..."}
```

## ⚙️ Configuration

To enable identity support, ensure the following file exists on your system:
- **Path**: `~/.odrl/did.json`
- **Format**:
```json
{
  "did": "did:oyd:your-unique-identifier-here",
  "keys": {
    "private_key": "..."
  }
}
```
If the file is missing, the server will continue to operate normally without DID attribution.

## 🧪 Testing

New tests have been added to verify ODRL functionality without interfering with the user's primary identity.

### 1. Run ODRL Unit Tests
Tests the lower-level identity loading logic using mocked home directories.
```bash
go test -v ./odrl/...
```

### 2. Run Server Integration Tests
Tests the API routes and response population.
```bash
go test -v ./server -run TestDIDHandler
```

### 3. Manual Verification
Start the server and query the identity:
```bash
./ollama serve &
curl http://localhost:11434/api/did
```

---
**Data Sovereignty**: *Ensuring that AI insights remain under the control of their originators via machine-readable rights and decentralized trust.*
