package opencodego

import (
	"github.com/goccy/go-json"

	"github.com/enterpilot/gomodel/internal/core"
	"github.com/enterpilot/gomodel/internal/providers/deepseek"
)

// reasoningReplayItem is the synthetic reasoning input item that stands in for
// a thinking turn the client never echoed back. Zen translates a text-bearing
// reasoning item into DeepSeek's reasoning_content; an item without content
// (summary only, or an id the upstream no longer knows) does not satisfy the
// thinking-mode rule.
var reasoningReplayItem = json.RawMessage(`{"type":"reasoning","summary":[],"content":[{"type":"reasoning_text","text":" "}]}`)

// padNativeResponsesReasoning restores the reasoning items DeepSeek requires on
// a request forwarded natively to /responses. Clients such as VS Code's unified
// chat provider replay a reasoning item only when it carries
// encrypted_content, which a DeepSeek turn never has, so history arrives bare
// and every turn after the first fails with "The reasoning_content in the
// thinking mode must be passed back to the API".
//
// The adaptation mirrors deepseek.padMissingReasoningContent on the chat path:
// it only applies when the request declares tools, because that is when the
// upstream enforces the rule, and it leaves assistant turns that already carry
// a reasoning item untouched.
func padNativeResponsesReasoning(req *core.ResponsesRequest) (*core.ResponsesRequest, error) {
	if req == nil || len(req.Tools) == 0 || !deepseek.IsModel(req.Model) {
		return req, nil
	}
	switch input := req.Input.(type) {
	case []core.ResponsesInputElement:
		padded := padResponsesElements(input)
		if padded == nil {
			return req, nil
		}
		copied := *req
		copied.Input = padded
		return &copied, nil
	case []map[string]any:
		items := make([]any, len(input))
		for i := range input {
			items[i] = input[i]
		}
		return copyWithPaddedResponsesInput(req, items)
	case []any:
		return copyWithPaddedResponsesInput(req, input)
	default:
		return req, nil
	}
}

func copyWithPaddedResponsesInput(req *core.ResponsesRequest, items []any) (*core.ResponsesRequest, error) {
	padded, ok := padResponsesAnyItems(items)
	if !ok {
		return req, nil
	}
	copied := *req
	copied.Input = padded
	return &copied, nil
}

// padResponsesElements walks decoded input items, inserting one reasoning item
// at the start of every assistant turn that has none. It returns nil when the
// input already satisfies the requirement, so the caller keeps the original.
func padResponsesElements(items []core.ResponsesInputElement) []core.ResponsesInputElement {
	out := make([]core.ResponsesInputElement, 0, len(items)+1)
	turnOpen, turnHasReasoning, inserted := false, false, false
	for _, item := range items {
		switch {
		case item.Type == "reasoning":
			turnHasReasoning = true
		case isAssistantResponseItem(item.Type, item.Role):
			if !turnOpen {
				if !turnHasReasoning {
					out = append(out, reasoningInputElement())
					inserted = true
				}
				turnOpen = true
			}
		default:
			turnOpen, turnHasReasoning = false, false
		}
		out = append(out, item)
	}
	if !inserted {
		return nil
	}
	return out
}

// padResponsesAnyItems is padResponsesElements for input decoded as generic
// maps rather than typed elements.
func padResponsesAnyItems(items []any) ([]any, bool) {
	out := make([]any, 0, len(items)+1)
	turnOpen, turnHasReasoning, inserted := false, false, false
	for _, item := range items {
		itemType, role := responsesItemKind(item)
		switch {
		case itemType == "reasoning":
			turnHasReasoning = true
		case isAssistantResponseItem(itemType, role):
			if !turnOpen {
				if !turnHasReasoning {
					out = append(out, reasoningInputElement())
					inserted = true
				}
				turnOpen = true
			}
		default:
			turnOpen, turnHasReasoning = false, false
		}
		out = append(out, item)
	}
	return out, inserted
}

// responsesItemKind reports the type and role of a generic input item. Items
// that are neither maps nor elements have no assistant turn of their own.
func responsesItemKind(item any) (string, string) {
	switch typed := item.(type) {
	case map[string]any:
		itemType, _ := typed["type"].(string)
		role, _ := typed["role"].(string)
		return itemType, role
	case core.ResponsesInputElement:
		return typed.Type, typed.Role
	default:
		return "", ""
	}
}

// isAssistantResponseItem reports whether an input item belongs to an assistant
// turn: an assistant message, or a function call that turn emitted.
func isAssistantResponseItem(itemType, role string) bool {
	if itemType == "function_call" {
		return true
	}
	return (itemType == "" || itemType == "message") && role == "assistant"
}

func reasoningInputElement() core.ResponsesInputElement {
	return core.ResponsesInputElement{Type: "reasoning", Raw: reasoningReplayItem}
}
