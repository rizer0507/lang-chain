"""Format instructions.

中文翻译:
格式说明。"""

JSON_FORMAT_INSTRUCTIONS = """STRICT OUTPUT FORMAT:
- Return only the JSON value that conforms to the schema. Do not include any additional text, explanations, headings, or separators.
- Do not wrap the JSON in Markdown or code fences (no ``` or ```json).
- Do not prepend or append any text (e.g., do not write "Here is the JSON:").
- The response must be a single top-level JSON value exactly as required by the schema (object/array/etc.), with no trailing commas or comments.

The output should be formatted as a JSON instance that conforms to the JSON schema below.

As an example, for the schema {{"properties": {{"foo": {{"title": "Foo", "description": "a list of strings", "type": "array", "items": {{"type": "string"}}}}}}, "required": ["foo"]}} the object {{"foo": ["bar", "baz"]}} is a well-formatted instance of the schema. The object {{"properties": {{"foo": ["bar", "baz"]}}}} is not well-formatted.

Here is the output schema (shown in a code block for readability only — do not include any backticks or Markdown in your output):
```
{schema}
```

 中文翻译:
 严格的输出格式：
- 仅返回符合模式的 JSON 值。请勿包含任何其他文本、解释、标题或分隔符。
- 不要将 JSON 包装在 Markdown 或代码围栏中（无````或````json）。
- 不要在前面或后面添加任何文本（例如，不要写“Here is the JSON:”）。
- 响应必须是完全符合架构（对象/数组/等）要求的单个顶级 JSON 值，没有尾随逗号或注释。
输出应格式化为符合以下 JSON 架构的 JSON 实例。
例如，对于架构 {{"properties": {{"foo": {{"title": "Foo", "description": "a list of strings", "type": "array", "items": {{"type": "string"}}}}}}, "required": ["foo"]}} 对象 {{"foo": ["bar", "baz"]}} 是架构的格式良好的实例。对象 {{"properties": {{"foo": ["bar", "baz"]}}}} 格式不正确。
以下是输出架构（仅出于可读性而在代码块中显示 - 不要在输出中包含任何反引号或 Markdown）：
````
{模式}
````"""  # noqa: E501
