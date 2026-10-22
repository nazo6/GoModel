// Package opencodego provides OpenCode Zen (Go subscription) integration for
// the LLM gateway.
package opencodego

import (
	"context"
	"io"
	"log/slog"
	"net/http"
	"os"
	"strings"

	"github.com/enterpilot/gomodel/internal/core"
	"github.com/enterpilot/gomodel/internal/providers"
	"github.com/enterpilot/gomodel/internal/providers/anthropic"
	"github.com/enterpilot/gomodel/internal/providers/deepseek"
	"github.com/enterpilot/gomodel/internal/providers/openai"
)

// defaultBaseURL is the OpenCode Zen "Go" endpoint. Its /chat/completions and
// /models routes are OpenAI-compatible (Bearer auth); its /messages route is
// Anthropic-native (x-api-key auth). Override with OPENCODE_GO_BASE_URL.
const defaultBaseURL = "https://opencode.ai/zen/go/v1"

// messagesModelsEnvVar names a comma-separated override list of model IDs that
// must be routed to the Anthropic-native /messages endpoint.
const messagesModelsEnvVar = "OPENCODE_GO_MESSAGES_MODELS"

// responsesModelsEnvVar names a comma-separated override list of model IDs
// that must be routed to the Responses-native /responses endpoint. When set,
// it replaces both default model lists entirely.
const responsesModelsEnvVar = "OPENCODE_GO_RESPONSES_MODELS"

// defaultMessagesModels lists the model IDs OpenCode Zen serves ONLY on its
// Anthropic-native /messages endpoint — /chat/completions rejects them with
// "not supported for format oa-compat". This is a temporary hardcoded set;
// upstream does not yet expose per-model endpoint metadata, so the split is
// maintained here and can be overridden with OPENCODE_GO_MESSAGES_MODELS until
// metadata-driven routing replaces it.
var defaultMessagesModels = []string{"qwen3.7-max"}

// defaultResponsesOnlyModels lists the model IDs OpenCode Zen serves ONLY on
// its Responses API — /chat/completions and /messages reject them with
// "ModelProtocolUnsupported". Like defaultMessagesModels this is a temporary
// hardcoded set, maintained here until upstream exposes per-model endpoint
// metadata. Chat-completions ingress for these models is not translated up to
// Responses yet (ENTERPILOT/GoModel#1084); until it is, they are reachable
// through /v1/responses.
var defaultResponsesOnlyModels = []string{"grok-4.7", "muse-spark-1.2-contributor", "muse-spark-1.3-contributor"}

// defaultDualSurfaceModels lists the model IDs OpenCode Zen serves on BOTH
// /chat/completions and /responses. Chat ingress keeps the chat surface;
// Responses ingress goes native. deepseek-v4.1-flash needs the native surface
// because Zen's chat path rejects non-string tool message content, which
// flattens image-bearing tool results arriving through /v1/responses.
var defaultDualSurfaceModels = []string{"deepseek-v4.1-flash"}

// Registration provides factory registration for the OpenCode Go provider.
// The "opencode_go" type derives OPENCODE_GO_API_KEY, OPENCODE_GO_BASE_URL, and
// OPENCODE_GO_MODELS by convention.
var Registration = providers.Registration{
	Type: "opencode_go",
	New:  New,
	Discovery: providers.DiscoveryConfig{
		DefaultBaseURL: defaultBaseURL,
	},
}

// messagesProvider is the subset of the Anthropic provider OpenCode Go reuses
// to translate canonical chat requests onto the Anthropic-native /messages
// endpoint.
type messagesProvider interface {
	ChatCompletion(context.Context, *core.ChatRequest) (*core.ChatResponse, error)
	StreamChatCompletion(context.Context, *core.ChatRequest) (io.ReadCloser, error)
}

// Provider implements the core.Provider interface for OpenCode Go.
//
// OpenCode Zen exposes three upstream dialects behind one base URL:
// OpenAI-style /chat/completions (most models), Anthropic-style /messages (a
// few models rejected by /chat/completions), and the Responses API (models
// rejected by both). The embedded ChatCompatible handles the former; chat
// requests for models in messagesModels are delegated to an Anthropic provider
// pinned to the same base URL, mirroring how GoModel already serves
// /v1/messages for native Anthropic; Responses requests for models in
// responsesModels are forwarded natively to /responses. Every path normalizes
// to the canonical OpenAI-shaped response, so callers see one consistent
// surface.
type Provider struct {
	*openai.ChatCompatible
	messages        messagesProvider
	messagesModels  map[string]struct{}
	responsesModels map[string]struct{}
	// headers yields the session/client identification headers (session.go).
	// Typed requests receive them through the CompatibleProvider SetHeaders
	// hook and the Anthropic request-headers hook; Passthrough merges them
	// under the caller's opaque headers.
	headers func(context.Context) http.Header
}

var _ core.Provider = (*Provider)(nil)

// New creates a new OpenCode Go provider.
func New(cfg providers.ProviderConfig, opts providers.ProviderOptions) core.Provider {
	baseURL := providers.ResolveBaseURL(cfg.BaseURL, defaultBaseURL)
	headers := requestHeaders(loadSessionHeaderEnabled())
	chat := openai.NewChatCompatible(cfg.APIKey, opts, compatibleConfig(baseURL, headers))
	// opts carries the shared keyring, so the /messages client rotates in step
	// with the chat client above rather than pinning the primary key.
	messages := anthropic.New(providers.ProviderConfig{APIKey: cfg.APIKey, APIKeys: cfg.APIKeys, BaseURL: baseURL}, opts)
	if native, ok := messages.(*anthropic.Provider); ok {
		native.SetRequestHeaders(headers)
	}
	messagesModels := loadMessagesModels()
	responsesModels := loadResponsesModels()
	// A model speaks exactly one upstream surface: until upstream exposes
	// per-model metadata, a misconfigured overlap would route the two ingress
	// directions to different dialects. Prefer the native Responses surface
	// and warn so the deployer can fix the lists.
	for id := range responsesModels {
		if _, ok := messagesModels[id]; ok {
			slog.Warn("model listed in both OpenCode Go surface lists; using the Responses surface",
				"model", id,
				"messages_env", messagesModelsEnvVar,
				"responses_env", responsesModelsEnvVar)
			delete(messagesModels, id)
		}
	}
	return &Provider{
		ChatCompatible:  chat,
		messages:        messages,
		messagesModels:  messagesModels,
		responsesModels: responsesModels,
		headers:         headers,
	}
}

// compatibleConfig describes the OpenAI-compatible half of the provider. The
// AdaptChatRequest hook carries OpenCode Zen's reasoning quirk (see
// reasoning.go) plus DeepSeek's rules for DeepSeek models (see deepseek.go).
// SetHeaders carries the bearer credential and the session/client
// identification headers (see session.go) on every typed request, so the
// native /responses path and the Responses-via-chat translation send them
// too; request-specific headers are applied after it and still win, keeping
// passthrough's caller-wins semantics.
func compatibleConfig(baseURL string, headers func(context.Context) http.Header) openai.CompatibleProviderConfig {
	return openai.CompatibleProviderConfig{
		ProviderName:     "opencode_go",
		BaseURL:          baseURL,
		AdaptChatRequest: chatRequestAdapter(loadDefaultReasoningEffort(), deepseek.LoadJSONSchemaMode()),
		SetHeaders: func(req *http.Request, apiKey string) {
			providers.SetAuthHeaders(req, apiKey, providers.AuthHeaderConfig{AuthScheme: "Bearer "})
			for name, values := range headers(req.Context()) {
				if !hasHeader(req.Header, name) {
					req.Header[http.CanonicalHeaderKey(name)] = values
				}
			}
		},
	}
}

// loadMessagesModels returns the set of model IDs routed to /messages, using
// the OPENCODE_GO_MESSAGES_MODELS override when present.
func loadMessagesModels() map[string]struct{} {
	return loadModelSet(messagesModelsEnvVar, defaultMessagesModels)
}

// loadResponsesModels returns the set of model IDs routed natively to
// /responses: the responses-only and dual-surface defaults, or the
// OPENCODE_GO_RESPONSES_MODELS override when present.
func loadResponsesModels() map[string]struct{} {
	if override := strings.TrimSpace(os.Getenv(responsesModelsEnvVar)); override != "" {
		return modelSet(strings.Split(override, ","))
	}
	set := modelSet(defaultResponsesOnlyModels)
	for _, id := range defaultDualSurfaceModels {
		set[id] = struct{}{}
	}
	return set
}

// modelSet builds the lookup set from trimmed, non-empty model IDs.
func modelSet(ids []string) map[string]struct{} {
	set := make(map[string]struct{}, len(ids))
	for _, id := range ids {
		if id = strings.TrimSpace(id); id != "" {
			set[id] = struct{}{}
		}
	}
	return set
}

// loadModelSet returns the model IDs in the comma-separated env override when
// present, and in defaults otherwise.
func loadModelSet(envVar string, defaults []string) map[string]struct{} {
	if override := strings.TrimSpace(os.Getenv(envVar)); override != "" {
		return modelSet(strings.Split(override, ","))
	}
	return modelSet(defaults)
}

// usesMessages reports whether a model must be served through the Anthropic
// /messages endpoint.
func (p *Provider) usesMessages(model string) bool {
	return inModelSet(p.messagesModels, model)
}

// usesResponses reports whether a model must be served through the native
// Responses API.
func (p *Provider) usesResponses(model string) bool {
	return inModelSet(p.responsesModels, model)
}

// inModelSet reports whether a model belongs to set. It matches on the bare
// model ID, tolerating an "opencode_go/" provider prefix.
func inModelSet(set map[string]struct{}, model string) bool {
	if i := strings.LastIndex(model, "/"); i >= 0 {
		model = model[i+1:]
	}
	_, ok := set[model]
	return ok
}

// ChatCompletion routes Anthropic-only models to /messages and everything else
// to /chat/completions.
func (p *Provider) ChatCompletion(ctx context.Context, req *core.ChatRequest) (*core.ChatResponse, error) {
	if req != nil && p.usesMessages(req.Model) {
		return p.messages.ChatCompletion(ctx, req)
	}
	return p.ChatCompatible.ChatCompletion(ctx, req)
}

// StreamChatCompletion mirrors ChatCompletion's per-model endpoint routing.
func (p *Provider) StreamChatCompletion(ctx context.Context, req *core.ChatRequest) (io.ReadCloser, error) {
	if req != nil && p.usesMessages(req.Model) {
		return p.messages.StreamChatCompletion(ctx, req)
	}
	return p.ChatCompatible.StreamChatCompletion(ctx, req)
}

// Responses serves each model on its native upstream surface: models in
// responsesModels are forwarded natively to /responses, and every other model
// keeps the chat-completions translation, which honors the per-model /messages
// routing through this provider's ChatCompletion.
func (p *Provider) Responses(ctx context.Context, req *core.ResponsesRequest) (*core.ResponsesResponse, error) {
	if req != nil && p.usesResponses(req.Model) {
		adapted, err := padNativeResponsesReasoning(req)
		if err != nil {
			return nil, err
		}
		return p.Compatible().Responses(ctx, adapted)
	}
	return providers.ResponsesViaChat(ctx, p, req, "opencode_go")
}

// StreamResponses mirrors Responses' per-model upstream surface routing.
func (p *Provider) StreamResponses(ctx context.Context, req *core.ResponsesRequest) (io.ReadCloser, error) {
	if req != nil && p.usesResponses(req.Model) {
		adapted, err := padNativeResponsesReasoning(req)
		if err != nil {
			return nil, err
		}
		return p.Compatible().StreamResponses(ctx, adapted)
	}
	return providers.StreamResponsesViaChat(ctx, p, req, "opencode_go")
}

// Passthrough forwards an opaque request with the identification headers
// filled in wherever the caller did not send its own. Passthrough is opaque
// by contract, so a caller-provided x-opencode-session, x-opencode-client, or
// User-Agent always wins over the gateway's value.
func (p *Provider) Passthrough(ctx context.Context, req *core.PassthroughRequest) (*core.PassthroughResponse, error) {
	if req == nil || p.headers == nil {
		return p.ChatCompatible.Passthrough(ctx, req)
	}
	merged := *req
	merged.Headers = withDefaultHeaders(req.Headers, p.headers(ctx))
	return p.ChatCompatible.Passthrough(ctx, &merged)
}

// withDefaultHeaders returns a copy of headers with every entry of defaults
// added that headers does not already carry. Names are compared
// case-insensitively so a non-canonical caller key still counts as present.
func withDefaultHeaders(headers, defaults http.Header) http.Header {
	merged := headers.Clone()
	if merged == nil {
		merged = make(http.Header, len(defaults))
	}
	for name, values := range defaults {
		if hasHeader(merged, name) {
			continue
		}
		merged[http.CanonicalHeaderKey(name)] = values
	}
	return merged
}

// hasHeader reports whether headers carries a non-empty entry for name under
// any casing of the key.
func hasHeader(headers http.Header, name string) bool {
	for key, values := range headers {
		if strings.EqualFold(key, name) && len(values) > 0 {
			return true
		}
	}
	return false
}

// Embeddings returns an error because OpenCode Go does not expose an embeddings endpoint.
func (p *Provider) Embeddings(_ context.Context, _ *core.EmbeddingRequest) (*core.EmbeddingResponse, error) {
	return nil, core.NewInvalidRequestError("opencode_go does not support embeddings", nil)
}
