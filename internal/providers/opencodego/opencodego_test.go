package opencodego

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/enterpilot/gomodel/internal/core"
	"github.com/enterpilot/gomodel/internal/llmclient"
	"github.com/enterpilot/gomodel/internal/providers"
	"github.com/enterpilot/gomodel/internal/providers/providertest"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// newTestProvider builds a provider whose OpenAI-compatible and Anthropic
// /messages paths both point at the same test server.
func newTestProvider(serverURL string, client *http.Client) *Provider {
	return newHTTPTestProvider("sk-opencode", serverURL, client, llmclient.Hooks{})
}

func TestChatCompatibleContract(t *testing.T) {
	// The contract model is on neither per-model surface, so the shared
	// chat/translation expectations hold; the native /responses surface for
	// responsesModels is covered by the routing tests below. Pin both surface
	// env overrides so an ambient override cannot reroute the contract model.
	t.Setenv(messagesModelsEnvVar, "contract-messages-model")
	t.Setenv(responsesModelsEnvVar, "contract-responses-model")
	providertest.AssertChatCompatible(t, providertest.ChatCompatible{
		Registration:   Registration,
		Type:           "opencode_go",
		DefaultBaseURL: "https://opencode.ai/zen/go/v1",
		New: func(apiKey, baseURL string, client *http.Client, hooks llmclient.Hooks) core.Provider {
			opts := providertest.Options(hooks)
			opts.HTTPClient = client
			return New(providers.ProviderConfig{APIKey: apiKey, BaseURL: baseURL}, opts)
		},
		Embeddings: false,
	})
	providertest.AssertNoNativeSurfaces(t, newTestProvider("", nil))
}

func TestChatCompletion_AnthropicStyleModel_UsesMessages(t *testing.T) {
	server, capture := providertest.JSONServer(t, http.StatusOK, `{
		"id":"msg_opencode",
		"model":"qwen3.7-max",
		"content":[{"type":"text","text":"hello"}],
		"stop_reason":"end_turn",
		"usage":{"input_tokens":5,"output_tokens":2}
	}`)

	resp, err := newTestProvider(server.URL, server.Client()).ChatCompletion(context.Background(), &core.ChatRequest{
		Model:    "qwen3.7-max",
		Messages: []core.Message{{Role: "user", Content: "hi"}},
	})
	require.NoError(t, err)

	req := capture.Last(t)
	assert.Equal(t, "/messages", req.Path)
	assert.Equal(t, "sk-opencode", req.Header.Get("x-api-key"))
	assert.Empty(t, req.Header.Get("Authorization"))
	assert.NotEmpty(t, req.Header.Get("anthropic-version"))
	require.Len(t, resp.Choices, 1)
	assert.Equal(t, "hello", resp.Choices[0].Message.Content)
}

func TestStreamChatCompletion_RoutesByModel(t *testing.T) {
	tests := []struct {
		name     string
		model    string
		wantPath string
	}{
		{"openai-style", "glm-5.1", "/chat/completions"},
		{"anthropic-style", "qwen3.7-max", "/messages"},
		{"prefixed anthropic-style", "opencode_go/qwen3.7-max", "/messages"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			server, capture := providertest.SSEServer(t, "data: [DONE]\n\n")

			stream, err := newTestProvider(server.URL, server.Client()).StreamChatCompletion(context.Background(), &core.ChatRequest{
				Model:    tt.model,
				Messages: []core.Message{{Role: "user", Content: "hi"}},
			})
			require.NoError(t, err)

			_, _ = io.Copy(io.Discard, stream)
			_ = stream.Close()
			assert.Equal(t, tt.wantPath, capture.Last(t).Path)
		})
	}
}

func TestMessagesModels_EnvOverride(t *testing.T) {
	t.Setenv(messagesModelsEnvVar, "foo-model, bar-model ")
	p := newTestProvider("", nil)

	assert.True(t, p.usesMessages("foo-model"))
	assert.True(t, p.usesMessages("bar-model"))
	assert.False(t, p.usesMessages("qwen3.7-max"))
	assert.True(t, p.usesMessages("opencode_go/foo-model"))
}

func TestResponsesModels_EnvOverride(t *testing.T) {
	t.Setenv(responsesModelsEnvVar, "foo-model, bar-model ")
	p := newTestProvider("", nil)

	assert.True(t, p.usesResponses("foo-model"))
	assert.True(t, p.usesResponses("bar-model"))
	assert.False(t, p.usesResponses("grok-4.7"))
	assert.True(t, p.usesResponses("opencode_go/foo-model"))
}

// TestNew_OverlappingSurfaceListsPreferResponses pins the overlap rule: a
// model listed on both surfaces is served on the Responses surface, and the
// /messages half drops it.
func TestNew_OverlappingSurfaceListsPreferResponses(t *testing.T) {
	t.Setenv(messagesModelsEnvVar, "dual-model")
	t.Setenv(responsesModelsEnvVar, "dual-model")
	p := newTestProvider("", nil)

	assert.True(t, p.usesResponses("dual-model"))
	assert.False(t, p.usesMessages("dual-model"))
}

// surfaceServer answers each upstream surface with a fixture of its own
// dialect, so one routing test can assert which path every model reaches.
func surfaceServer(t *testing.T) (*httptest.Server, *providertest.Capture) {
	t.Helper()
	return providertest.Server(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		switch r.URL.Path {
		case "/responses":
			_, _ = w.Write([]byte(providertest.ResponsesJSON))
		case "/messages":
			_, _ = w.Write([]byte(`{"id":"msg_opencode","type":"message","role":"assistant","model":"qwen3.7-max","content":[{"type":"text","text":"hello"}],"stop_reason":"end_turn","usage":{"input_tokens":5,"output_tokens":2}}`))
		default:
			_, _ = w.Write([]byte(providertest.ChatCompletionJSON))
		}
	})
}

// TestResponses_RoutesByModel pins the per-model upstream surface of the
// Responses ingress: Responses-only models are forwarded natively to
// /responses while every other model keeps the chat-completions translation
// (/messages included).
func TestResponses_RoutesByModel(t *testing.T) {
	t.Setenv(messagesModelsEnvVar, "qwen3.7-max")
	t.Setenv(responsesModelsEnvVar, "")
	server, capture := surfaceServer(t)

	tests := []struct {
		name     string
		model    string
		wantPath string
	}{
		{"responses-native", "grok-4.7", "/responses"},
		{"prefixed responses-native", "opencode_go/muse-spark-1.3-contributor", "/responses"},
		{"openai-style", "glm-5.1", "/chat/completions"},
		{"anthropic-style", "qwen3.7-max", "/messages"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			resp, err := newTestProvider(server.URL, server.Client()).Responses(context.Background(), &core.ResponsesRequest{
				Model: tt.model,
				Input: "hi",
			})
			require.NoError(t, err)
			require.NotNil(t, resp)

			req := capture.Last(t)
			assert.Equal(t, tt.wantPath, req.Path)
			sent := req.JSON(t)
			assert.Equal(t, tt.model, sent["model"])
			if tt.wantPath == "/responses" {
				assert.Equal(t, "hi", sent["input"], "native requests keep the Responses input")
			}
		})
	}
}

// TestResponses_PreservesArrayFormToolOutput pins the multimodal tool-result
// contract on the native /responses path: array-form function_call_output
// outputs (input_image parts, as VS Code and other clients return screenshots)
// are forwarded verbatim instead of being flattened into base64 text, which
// upstream tokenizers would count as message tokens (context-length 400s).
func TestResponses_PreservesArrayFormToolOutput(t *testing.T) {
	t.Setenv(messagesModelsEnvVar, "")
	t.Setenv(responsesModelsEnvVar, "")
	server, capture := surfaceServer(t)

	resp, err := newTestProvider(server.URL, server.Client()).Responses(context.Background(), &core.ResponsesRequest{
		Model: "grok-4.7",
		Input: []core.ResponsesInputElement{
			{Type: "function_call_output", CallID: "call_1",
				Output: json.RawMessage(`[{"type":"input_image","image_url":"data:image/png;base64,AAAA"}]`)},
		},
	})
	require.NoError(t, err)
	require.NotNil(t, resp)

	sent := capture.Last(t).JSON(t)
	input, ok := sent["input"].([]any)
	require.True(t, ok, "input = %#v, want an array", sent["input"])
	require.Len(t, input, 1)
	item, ok := input[0].(map[string]any)
	require.True(t, ok)
	require.Equal(t, "function_call_output", item["type"])
	require.Equal(t, "call_1", item["call_id"])
	output, ok := item["output"].([]any)
	require.True(t, ok, "output = %#v, want an array on the wire, not stringified base64", item["output"])
	require.Len(t, output, 1)
	part, ok := output[0].(map[string]any)
	require.True(t, ok)
	require.Equal(t, "input_image", part["type"])
	require.Equal(t, "data:image/png;base64,AAAA", part["image_url"])
}

// TestStreamResponses_RoutesByModel mirrors the non-streaming routing test for
// streaming Responses requests.
func TestStreamResponses_RoutesByModel(t *testing.T) {
	t.Setenv(messagesModelsEnvVar, "qwen3.7-max")
	t.Setenv(responsesModelsEnvVar, "")
	server, capture := providertest.SSEServer(t, "data: [DONE]\n\n")

	tests := []struct {
		name     string
		model    string
		wantPath string
	}{
		{"responses-native", "muse-spark-1.2-contributor", "/responses"},
		{"openai-style", "glm-5.1", "/chat/completions"},
		{"anthropic-style", "qwen3.7-max", "/messages"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			stream, err := newTestProvider(server.URL, server.Client()).StreamResponses(context.Background(), &core.ResponsesRequest{
				Model: tt.model,
				Input: "hi",
			})
			require.NoError(t, err)

			_, _ = io.Copy(io.Discard, stream)
			_ = stream.Close()
			assert.Equal(t, tt.wantPath, capture.Last(t).Path)
		})
	}
}
