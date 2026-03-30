# ODRL Support in Ollama

This document describes the integration of **ODRL (Open Digital Rights Language)** and **Decentralized Identifiers (DIDs)** within the Ollama ecosystem. These features enable data sovereignty, verifiable provenance, and explicit attribution for model outputs.

## 🛡️ Feature Overview

Ollama now supports identifying instances via **DIDs**. By linking an Ollama instance to a DID, every generation, chat response, and model listing can be cryptographically attributed to its owner. This provides a foundation for enforcing usage policies (ODRL Agreements) in distributed AI environments.

### Key Capabilities:
- **Server Identity**: The server discover its identity from a local ODRL wallet (`~/.odrl/did.json`).
- **Attribution**: Generation and Chat responses include the server's DID for transparency.
- **Discovery**: Clients can query the server's identity via the `/api/did` endpoint.
- **Model Metadata**: Listings and process statuses now include the identity of the serving instance.

## 🔍 How ODRL Support Works

The ODRL integration follows a **zero-trust** and **sovereignty-first** architecture. The core mechanism facilitates the linking of computational output to a verifiable identity without requiring a central authority.

### The Identity Flow:
1.  **Wallet Discovery**: Upon startup or request, the Ollama server probes the local environment for an ODRL wallet located at `~/.odrl/did.json`.
2.  **DID Resolution**: The `odrl` package extracts the `did` string. If present, this DID becomes the "Active Identity" of the server instance.
3.  **Automatic Attribution**: For every request processed through the `/api/chat` or `/api/generate` pipelines, the server injects the `did` field into the final response JSON.
4.  **Provenance Chain**: By including the DID in the response metadata, any consumer of the model's output can verify which specific instance produced the data. This allows for:
    *   **Auditability**: Tracking which servers performed specific inferences.
    *   **Agreement Compliance**: Ensuring that the serving instance has the rights to serve the model as defined in an ODRL Agreement (linking the Asset DID to the Server DID).

## 🆔 DID Identifier Creation

DIDs used in this project areTypically created using the **ODRL Expert Skill** and the **CODATA ODRL Infrastructure**.

### Creation Process:
1.  **Generation**: A new DID is generated via a secure request to the ODRL API. The process uses a unique seed and agent identification (e.g., "Antigravity AI") to ensure uniqueness.
2.  **Key Package**: The API returns a DID Document and a corresponding key package containing the `private_key` and `revocation_key`.
3.  **Local Storage**: The resulting JSON is saved to `~/.odrl/did.json`. This file serves as the server's "identity card."

### Using the ODRL Expert Skill:
If you have the **Croissant Toolkit** installed, you can create a new identity using the following command:
```bash
python3 .gemini/skills/odrl-expert/scripts/odrl_client.py init
```
This command checks for an existing identity and, if none is found, registers a new **OOYDID** (Own Your Data DID) on the global infrastructure.

### DID Format:
A typical DID looks like this:
`did:oyd:zQmcVHWDMeXtj273A9gNAnEG2EdrGEjtQiFuw9PncyVgs9z`
- `did`: The standard URI scheme.
- `oyd`: The method identifier (Own Your Data).
- `zQmc...`: A unique cryptographic hash representing the public identity document.

## 🛠️ Implementation Details

### 1. The `odrl` Package
A new internal package `github.com/ollama/ollama/odrl` handles the discovery and loading of the instance identity.

- **Storage**: Identity is read from `~/.odrl/did.json`.
- **Primary Method**: `odrl.GetDID()` returns the master DID as a string.

### 2. API Extensions
New fields have been added to the following API response structs:
- **Generate**: Includes `did` in the JSON response.
- **Chat**: Includes `did` in the JSON response.
- **Tags & List**: Includes `did` for every model.
- **Process (ps)**: Shows which identity is running a model.
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
If the file is missing, the server will operate normally without DID attribution.

## 🧪 Testing

New tests verify ODRL functionality without interfering with your primary identity.

### 1. Run ODRL Unit Tests
Tests the lower-level identity loading logic using mocked environments.
```bash
go test -v ./odrl/...
```

### 2. Run Server Integration Tests
Tests the API routes and response population.
```bash
go test -v ./server -run TestDIDHandler
```

---
**Data Sovereignty**: *Ensuring that AI insights remain under the control of their originators via machine-readable rights and decentralized trust.*
