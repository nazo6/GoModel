package opencodego

import (
	"context"
	"encoding/json"
	"io"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/enterpilot/gomodel/internal/core"
	"github.com/enterpilot/gomodel/internal/providers/providertest"
)

// lookupTool is the minimal tool declaration the thinking-mode rule needs:
// DeepSeek only enforces reasoning replay on tools-bearing requests.
var lookupTool = []map[string]any{
	{"type": "function", "name": "lookup_weather", "parameters": map[string]any{"type": "object"}},
}

// TestResponses_NativeDeepSeekPadsMissingReasoning pins the fix for DeepSeek's
// thinking-mode replay rule on the native /responses path: a client that echoes
// no reasoning item (VS Code replays one only when it carries
// encrypted_content) still produces a history the upstream accepts.
func TestResponses_NativeDeepSeekPadsMissingReasoning(t *testing.T) {
	t.Setenv(responsesModelsEnvVar, "")
	server, capture := surfaceServer(t)

	tests := []struct {
		name  string
		input any
		// wantItems is the expected item-type sequence after the user message.
		wantItems []string
		// wantReasoningAt lists the indexes holding the injected reasoning item.
		wantReasoningAt []int
	}{
		{
			name: "bare tool-call turn (typed input)",
			input: []core.ResponsesInputElement{
				{Type: "message", Role: "user", Content: "weather?"},
				{Type: "function_call", CallID: "call_1", Name: "lookup_weather", Arguments: "{}"},
				{Type: "function_call_output", CallID: "call_1", Output: "sunny"},
			},
			wantItems:       []string{"user", "reasoning", "function_call", "function_call_output"},
			wantReasoningAt: []int{1},
		},
		{
			name: "assistant message then tool call (typed input)",
			input: []core.ResponsesInputElement{
				{Type: "message", Role: "user", Content: "weather?"},
				{Type: "message", Role: "assistant", Content: "Checking."},
				{Type: "function_call", CallID: "call_1", Name: "lookup_weather", Arguments: "{}"},
				{Type: "function_call_output", CallID: "call_1", Output: "sunny"},
			},
			wantItems:       []string{"user", "reasoning", "assistant", "function_call", "function_call_output"},
			wantReasoningAt: []int{1},
		},
		{
			name: "two bare tool rounds (typed input)",
			input: []core.ResponsesInputElement{
				{Type: "message", Role: "user", Content: "weather?"},
				{Type: "function_call", CallID: "call_1", Name: "lookup_weather", Arguments: "{}"},
				{Type: "function_call_output", CallID: "call_1", Output: "sunny"},
				{Type: "function_call", CallID: "call_2", Name: "lookup_weather", Arguments: "{}"},
				{Type: "function_call_output", CallID: "call_2", Output: "rainy"},
			},
			wantItems:       []string{"user", "reasoning", "function_call", "function_call_output", "reasoning", "function_call", "function_call_output"},
			wantReasoningAt: []int{1, 4},
		},
		{
			name: "bare tool-call turn (generic maps)",
			input: []any{
				map[string]any{"type": "message", "role": "user", "content": "weather?"},
				map[string]any{"type": "function_call", "call_id": "call_1", "name": "lookup_weather", "arguments": "{}"},
				map[string]any{"type": "function_call_output", "call_id": "call_1", "output": "sunny"},
			},
			wantItems:       []string{"user", "reasoning", "function_call", "function_call_output"},
			wantReasoningAt: []int{1},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := newTestProvider(server.URL, server.Client()).Responses(context.Background(), &core.ResponsesRequest{
				Model: "deepseek-v4.1-flash",
				Input: tt.input,
				Tools: lookupTool,
			})
			require.NoError(t, err)

			items := responseInputItems(t, capture.Last(t))
			require.Len(t, items, len(tt.wantItems))
			for i, want := range tt.wantItems {
				assert.Equal(t, want, responseItemLabel(items[i]), "input[%d] = %#v", i, items[i])
			}

			reasoningAt := make([]int, 0, 1)
			for i, item := range items {
				if item["type"] != "reasoning" {
					continue
				}
				reasoningAt = append(reasoningAt, i)
				assert.Equal(t, []any{}, item["summary"], "injected reasoning item keeps an empty summary")
				assert.Equal(t,
					[]any{map[string]any{"type": "reasoning_text", "text": " "}},
					item["content"],
					"injected reasoning item carries the neutral text the chat path pads",
				)
			}
			assert.Equal(t, tt.wantReasoningAt, reasoningAt)
		})
	}
}

// TestResponses_NativeDeepSeekKeepsClientReasoning pins that an assistant turn
// the client did replay reasoning for gains no second reasoning item.
func TestResponses_NativeDeepSeekKeepsClientReasoning(t *testing.T) {
	t.Setenv(responsesModelsEnvVar, "")
	server, capture := surfaceServer(t)

	input := []core.ResponsesInputElement{
		{Type: "message", Role: "user", Content: "weather?"},
		{Type: "reasoning", Raw: []byte(`{"type":"reasoning","summary":[],"content":[{"type":"reasoning_text","text":"client thinking"}]}`)},
		{Type: "function_call", CallID: "call_1", Name: "lookup_weather", Arguments: "{}"},
		{Type: "function_call_output", CallID: "call_1", Output: "sunny"},
	}
	_, err := newTestProvider(server.URL, server.Client()).Responses(context.Background(), &core.ResponsesRequest{
		Model: "deepseek-v4.1-flash",
		Input: input,
		Tools: lookupTool,
	})
	require.NoError(t, err)

	items := responseInputItems(t, capture.Last(t))
	require.Len(t, items, len(input), "client reasoning must not be duplicated")
	require.Equal(t, "reasoning", items[1]["type"])
	content, ok := items[1]["content"].([]any)
	require.True(t, ok)
	require.Len(t, content, 1)
	assert.Equal(t, "client thinking", content[0].(map[string]any)["text"])
}

// TestResponses_NativeReasoningPaddingSkipped pins when the adaptation stays
// out of the way: non-DeepSeek models, requests without tools (the upstream
// does not enforce the rule), and non-array input.
func TestResponses_NativeReasoningPaddingSkipped(t *testing.T) {
	t.Setenv(responsesModelsEnvVar, "")
	server, capture := surfaceServer(t)

	tests := []struct {
		name  string
		model string
		input any
		tools []map[string]any
	}{
		{
			name:  "non-deepseek model",
			model: "grok-4.7",
			input: []core.ResponsesInputElement{
				{Type: "message", Role: "user", Content: "weather?"},
				{Type: "function_call", CallID: "call_1", Name: "lookup_weather", Arguments: "{}"},
				{Type: "function_call_output", CallID: "call_1", Output: "sunny"},
			},
			tools: lookupTool,
		},
		{
			name:  "deepseek model without tools",
			model: "deepseek-v4.1-flash",
			input: []core.ResponsesInputElement{
				{Type: "message", Role: "user", Content: "weather?"},
				{Type: "message", Role: "assistant", Content: "sunny"},
			},
		},
		{
			name:  "string input",
			model: "deepseek-v4.1-flash",
			input: "hi",
			tools: lookupTool,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := newTestProvider(server.URL, server.Client()).Responses(context.Background(), &core.ResponsesRequest{
				Model: tt.model,
				Input: tt.input,
				Tools: tt.tools,
			})
			require.NoError(t, err)

			sent := capture.Last(t).JSON(t)
			assert.Equal(t, normalizeJSON(t, tt.input), sent["input"], "input must be forwarded untouched")
		})
	}
}

// TestStreamResponses_NativeDeepSeekPadsMissingReasoning mirrors the
// non-streaming test: both entry points forward through the same adaptation.
func TestStreamResponses_NativeDeepSeekPadsMissingReasoning(t *testing.T) {
	t.Setenv(responsesModelsEnvVar, "")
	server, capture := providertest.SSEServer(t, "data: [DONE]\n\n")

	stream, err := newTestProvider(server.URL, server.Client()).StreamResponses(context.Background(), &core.ResponsesRequest{
		Model: "deepseek-v4.1-flash",
		Input: []core.ResponsesInputElement{
			{Type: "message", Role: "user", Content: "weather?"},
			{Type: "function_call", CallID: "call_1", Name: "lookup_weather", Arguments: "{}"},
			{Type: "function_call_output", CallID: "call_1", Output: "sunny"},
		},
		Tools: lookupTool,
	})
	require.NoError(t, err)
	_, _ = io.Copy(io.Discard, stream)
	require.NoError(t, stream.Close())

	items := responseInputItems(t, capture.Last(t))
	require.Len(t, items, 4)
	assert.Equal(t, "reasoning", items[1]["type"])
}

// responseInputItems decodes the wire input array of a recorded request.
func responseInputItems(t *testing.T, req providertest.Recorded) []map[string]any {
	t.Helper()
	sent := req.JSON(t)
	raw, ok := sent["input"].([]any)
	require.True(t, ok, "input = %#v, want an array", sent["input"])
	items := make([]map[string]any, 0, len(raw))
	for _, item := range raw {
		decoded, ok := item.(map[string]any)
		require.True(t, ok, "input item = %#v, want an object", item)
		items = append(items, decoded)
	}
	return items
}

// responseItemLabel names an input item for sequence assertions.
func responseItemLabel(item map[string]any) string {
	if item["type"] == "message" || item["type"] == nil {
		role, _ := item["role"].(string)
		return role
	}
	name, _ := item["type"].(string)
	return name
}

// normalizeJSON round-trips a value through JSON so a typed input and its wire
// form compare equal.
func normalizeJSON(t *testing.T, value any) any {
	t.Helper()
	encoded, err := json.Marshal(value)
	require.NoError(t, err)
	var decoded any
	require.NoError(t, json.Unmarshal(encoded, &decoded))
	return decoded
}
