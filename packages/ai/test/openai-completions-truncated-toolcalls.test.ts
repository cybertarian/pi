import { beforeEach, describe, expect, it, vi } from "vitest";
import { stream as streamOpenAICompletions } from "../src/api/openai-completions.ts";
import type { Model } from "../src/types.ts";
import { normalizeContext } from "../src/utils/transcript.ts";

type ToolCallsDelta = Array<{
	index: number;
	id?: string;
	type?: string;
	function?: { name?: string; arguments?: string };
}>;

type Chunk = {
	id: string;
	choices: Array<{
		index: number;
		delta: { tool_calls?: ToolCallsDelta };
		finish_reason?: string | null;
	}>;
};

const mockState = vi.hoisted(() => ({
	chunks: [] as Chunk[],
}));

vi.mock("openai", () => {
	class FakeOpenAI {
		chat = {
			completions: {
				create: (_params: unknown, _options: unknown) => {
					const chunks = mockState.chunks;
					const stream = {
						async *[Symbol.asyncIterator]() {
							for (const chunk of chunks) {
								yield chunk;
							}
						},
					};
					const promise = Promise.resolve(stream) as Promise<typeof stream> & {
						withResponse: () => Promise<{
							data: typeof stream;
							response: { status: number; headers: Headers };
						}>;
					};
					promise.withResponse = async () => ({
						data: stream,
						response: { status: 200, headers: new Headers() },
					});
					return promise;
				},
			},
		};
	}
	return { default: FakeOpenAI };
});

const model: Model<"openai-completions"> = {
	id: "test-model",
	name: "Test Model",
	api: "openai-completions",
	provider: "openai",
	baseUrl: "https://api.openai.com/v1",
	reasoning: false,
	input: ["text"],
	cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
	contextWindow: 1000,
	maxTokens: 100,
};

const context = normalizeContext({
	systemPrompt: "",
	messages: [{ role: "user", content: [{ type: "text", text: "hi" }], timestamp: 0 }],
	tools: [],
});

async function consume() {
	const stream = streamOpenAICompletions(model, context, { apiKey: "test" });
	for await (const _event of stream) {
		void _event;
	}
	return stream.result();
}

function toolCalls(message: Awaited<ReturnType<typeof consume>>) {
	return message.content.filter((block) => block.type === "toolCall");
}

describe("openai-completions truncated stream tool-call handling", () => {
	beforeEach(() => {
		mockState.chunks = [];
	});

	it("drops a tool call that received no argument deltas before the stream ended without finish_reason", async () => {
		// Provider drops the connection right after the tool call starts:
		// the block exists but its arguments never stream (partialArgs stays empty).
		mockState.chunks = [
			{
				id: "chatcmpl-test",
				choices: [
					{
						index: 0,
						delta: {
							tool_calls: [{ index: 0, id: "call_1", type: "function", function: { name: "write" } }],
						},
					},
				],
			},
		];

		const result = await consume();

		expect(result.stopReason).toBe("error");
		expect(result.errorMessage).toContain("Stream ended without finish_reason");
		expect(toolCalls(result)).toHaveLength(0);
	});

	it("drops a tool call with partially streamed arguments when the stream ends without finish_reason", async () => {
		mockState.chunks = [
			{
				id: "chatcmpl-test",
				choices: [
					{
						index: 0,
						delta: {
							tool_calls: [{ index: 0, id: "call_1", type: "function", function: { name: "write" } }],
						},
					},
				],
			},
			{
				id: "chatcmpl-test",
				choices: [
					{
						index: 0,
						delta: {
							tool_calls: [{ index: 0, function: { arguments: '{"path":"a.md",' } }],
						},
					},
				],
			},
		];

		const result = await consume();

		expect(result.stopReason).toBe("error");
		expect(result.errorMessage).toContain("Stream ended without finish_reason");
		// The half-streamed arguments must not persist as a bogus tool call.
		expect(toolCalls(result)).toHaveLength(0);
	});

	it("keeps a tool call whose arguments fully streamed as valid JSON even if finish_reason never arrived", async () => {
		mockState.chunks = [
			{
				id: "chatcmpl-test",
				choices: [
					{
						index: 0,
						delta: {
							tool_calls: [
								{
									index: 0,
									id: "call_1",
									type: "function",
									function: { name: "write", arguments: '{"path":"a.md"}' },
								},
							],
						},
					},
				],
			},
		];

		const result = await consume();

		expect(result.stopReason).toBe("error");
		const calls = toolCalls(result);
		expect(calls).toHaveLength(1);
		const call = calls[0];
		if (!call || call.type !== "toolCall") throw new Error("Expected one toolCall");
		expect(call.name).toBe("write");
		expect(call.arguments).toEqual({ path: "a.md" });
		// No streaming scratch fields may leak into the persisted message.
		expect("partialArgs" in call).toBe(false);
		expect("argsComplete" in call).toBe(false);
		expect("customInput" in call).toBe(false);
	});

	it("keeps completed tool calls and strips scratch fields on the normal finish_reason path", async () => {
		mockState.chunks = [
			{
				id: "chatcmpl-test",
				choices: [
					{
						index: 0,
						delta: {
							tool_calls: [
								{
									index: 0,
									id: "call_1",
									type: "function",
									function: { name: "write", arguments: '{"path":"a.md"}' },
								},
							],
						},
					},
				],
			},
			{
				id: "chatcmpl-test",
				choices: [{ index: 0, delta: {}, finish_reason: "tool_calls" }],
			},
		];

		const result = await consume();

		expect(result.stopReason).toBe("toolUse");
		const calls = toolCalls(result);
		expect(calls).toHaveLength(1);
		const call = calls[0];
		if (!call || call.type !== "toolCall") throw new Error("Expected one toolCall");
		expect(call.arguments).toEqual({ path: "a.md" });
		expect("partialArgs" in call).toBe(false);
		expect("argsComplete" in call).toBe(false);
	});
});
