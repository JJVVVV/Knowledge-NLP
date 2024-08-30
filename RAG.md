## 关于trunk

文本分块是将长文本切分成小片段的过程，比如将一篇长文章切分成一个个相对短的段落。那么为什么要进行文本分块？**一方面当前** **LLM** **的上下文长度是有限制的**，直接把一篇长文全部作为相关信息放到 LLM 的上下文窗口中，可能会超过长度限制。**另一方面，对于长文本来说，即使其和查询的问题相关，但一般不会通篇都是完全相关的，而分块能一定程度上剔除不相关的内容**，为后续的回复生成过滤一些不必要的噪声。 文本分块的好坏将很大程度上影响后续回复生成的效果，切分得不好，内容之间的关联性会被切断。因此设计一个好的分块策略十分重要。**分块策略包括具体的切分方法 ( 比如是按句子切分还是段落切分 )，块的大小设为多少合适，不同的块之间是否允许重叠等**。Pinecone 的这篇博客 **[Chunking Strategies for LLM Applications](****[https://www.pinecone.io/learn/chunking-strategies/](https://link.zhihu.com/?target=https%3A//www.pinecone.io/learn/chunking-strategies/)****)**中就给出了一些在设计分块策略时需要考虑的因素。

-  原始内容的特点：原始内容是长文 ( 博客文章、书籍等 ) 还是短文 ( 推文、即时消息等 )，是什么格式 ( HTML、Markdown、Code 还是 LaTeX 等 )，不同的内容特点可能会适用不同的分块策略；
- 后续使用的索引方法：目前最常用的索引是对分块后的内容进行向量索引，那么不同的向量嵌入模型可能有其适用的分块大小，比如 sentence-transformer 模型比较适合对句子级别的内容进行嵌入，OpenAI 的 text-embedding-ada-002 模型比较适合的分块大小在 256~512 个标记数量；
- 问题的长度：问题的长度需要考虑，因为需要基于问题去检索出相关的文本片段；
- 检索出的相关内容在回复生成阶段的使用方法：如果是直接把检索出的相关内容作为 Prompt 的一部分提供给 LLM，那么 LLM 的输入长度限制在设计分块大小时就需要考虑。

**那么文本分块具体如何实现？一般来说，实现文本分块的整体流程如下:** 将原始的长文本切分成小的语义单元，这里的语义单元通常是句子级别或者段落级别； 将这些小的语义单元融合成更大的块，直到达到设定的块大小 ( Chunk Size )，就将该块作为独立的文本片段； 迭代构建下一个文本片段，一般相邻的文本片段之间会设置重叠，以保持语义的连贯性。 那如何把原始的长文本切分成小的语义单元? 最常用的是基于分割符进行切分，比如句号 ( . )、换行符 ( \\n )、空格等。除了可以利用单个分割符进行简单切分，还可以定义一组分割符进行迭代切分，比如定义 `["\n\n", "\n", " ", ""]` 这样一组分隔符，切分的时候先利用第一个分割符进行切分 ( 实现类似按段落切分的效果 )，第一次切分完成后，对于超过预设大小的块，继续使用后面的分割符进行切分，依此类推。这种切分方法能比较好地保持原始文本的层次结构。 **对于一些结构化的文本，比如代码，****Markdown****，LaTeX 等文本，在进行切分的时候可能需要单独进行考虑:** 比如 Python 代码文件，分割符中可能就需要加入类似 `\nclass `，`\ndef ` 这种来保证类和函数代码块的完整性； 比如 Markdown 文件，是通过不同层级的 Header 进行组织的，即不同数量的 \# 符号，在切分时就可以通过使用特定的分割符来维持这种层级结构。 文本块大小的设定也是分块策略需要考虑的重要因素，太大或者太小都会影响最终回复生成的效果。文本块大小的计算方法，最常用的可以直接基于字符数进行统计 ( Character-level )，也可以基于标记数进行统计 ( Token-level )。至于如何确定合适的分块大小，这个因场景而异，很难有一个统一的标准，可以通过评估不同分块大小的效果来进行选择。 上面提到的一些分块方法在 [LangChain]([https://python.langchain.com/docs/modules/data_connection/document_transformers/](https://link.zhihu.com/?target=https%3A//python.langchain.com/docs/modules/data_connection/document_transformers/))<sup>[12]</sup> 中都有相应的实现。比如下面的代码示例

```Python
from langchain.text_splitter import CharacterTextSplitterfrom langchain.text_splitter import RecursiveCharacterTextSplitter, Language# text splittext_splitter = RecursiveCharacterTextSplitter(# Set a really small chunk size, just to show.chunk_size = 100,chunk_overlap  = 20,length_function = len,add_start_index = True,)# code splitpython_splitter = RecursiveCharacterTextSplitter.from_language(language=Language.PYTHON, 
 chunk_size=50, 
 chunk_overlap=0 
)# markdown splitmd_splitter = RecursiveCharacterTextSplitter.from_language(  
 language=Language.MARKDOWN, 
 chunk_size=60, 
 chunk_overlap=0 
)
```

**个人想法**

对passage切分为trunk后，用LLM对trunk进行后处理，例如让LLM根据原始的整个passage对当前trunk进行指代消解，补充上下文关键信息来辅助理解trunk中的关键词的含义（例如trunk中提到苹果，根据整个passage补充上 "苹果是电脑品牌名，又叫Mac。。。。"）。