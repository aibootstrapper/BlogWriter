from typing import Dict, List, Optional, Any, Mapping

from langchain import LLMChain, PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import BaseLLM
from langchain.vectorstores.base import VectorStore
from pydantic import BaseModel, Field
from langchain.chains.base import Chain

from langchain.vectorstores import FAISS
from langchain.docstore import InMemoryDocstore


def get_vectorstore():
    # Define your embedding model
    embeddings_model = OpenAIEmbeddings()
    # Initialize the vectorstore as empty
    import faiss

    embedding_size = 1536
    index = faiss.IndexFlatL2(embedding_size)
    vectorstore = FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {})

    return vectorstore


def make_user_select_value(values: Mapping[str, str]) -> str:
    """Make a user select value."""
    print("Please select one of the following options:")
    for key, value in values.items():
        print(f"{key}. {value}")
    while True:
        print("Type none if none of the options are suitable.")
        selected_key = input("Enter the number of the selected option: ")
        if selected_key.lower() == "none":
            return None
        try:
            return values[selected_key]
        except KeyError:
            print("Invalid option selected. Please try again.")


def ask_human_for_feedback(prompt: str, content: str) -> str:
    """Edit the outline."""
    print(prompt)
    print(content)
    while True:
        make_changes = input("Do you want to make any changes? (y/n) ")
        if make_changes == "y":
            changes = input("Enter the changes you want to make: ")
            return changes
        elif make_changes == "n":
            return None
        else:
            print("Invalid option selected. Please try again.")


class BlogIdeationChain(LLMChain):
    """Chain to generates tasks."""

    @classmethod
    def from_llm(cls, llm: BaseLLM, verbose: bool = True) -> LLMChain:
        """Get the response parser."""
        topic_ideation_template = (
            "Generate 5 original and marketable blog topic ideas based on the given topic and keywords."
            "Ensure that the ideas are engaging, relevant, and optimized for SEO."
            "Topic: {topic}"
            "Keywords: {keywords}"
            "Remember to use the provided keywords in the blog topic while maintaining a natural and engaging tone."
            "Return the tasks as a json with the id as keys."
        )
        prompt = PromptTemplate(
            template=topic_ideation_template,
            input_variables=[
                "topic",
                "keywords",
            ],
        )
        return cls(prompt=prompt, llm=llm, verbose=verbose)

    def _parse_result(self, result: str):
        """Parse the result."""
        import json

        return json.loads(result)

    def run(self, **kwargs):
        """Run the chain."""
        result = super().run(**kwargs)
        return self._parse_result(result)


class OutlineWriterChain(LLMChain):
    """Chain to prioritize tasks."""

    @classmethod
    def from_llm(cls, llm: BaseLLM, verbose: bool = True) -> LLMChain:
        """Get the response parser."""
        outline_writer_template = (
            "Please create a well-structured and engaging blog outline based on the given blog topic and keywords."
            "Ensure that the outline is engaging, relevant, and optimized for SEO."
            "Topic: {topic}"
            "Keywords: {keywords}"
            "Remember to use the provided keywords in the blog topic while maintaining a natural and engaging tone."
        )
        prompt = PromptTemplate(
            template=outline_writer_template,
            input_variables=["topic", "keywords"],
        )
        return cls(prompt=prompt, llm=llm, verbose=verbose)


class OutlineParserChain(LLMChain):
    @classmethod
    def from_llm(cls, llm: BaseLLM, verbose: bool = True) -> LLMChain:
        outline_parser_template = (
            "Read the following outline and parse it into a JSON object with headings as keys to an array of subheadings."
            "Outline: {outline}"
        )
        prompt = PromptTemplate(
            template=outline_parser_template,
            input_variables=["outline"],
        )
        return cls(prompt=prompt, llm=llm, verbose=verbose)

    def _parse_result(self, result: str):
        """Parse the result."""
        import json

        return json.loads(result)

    def run(self, **kwargs):
        """Run the chain."""
        result = super().run(**kwargs)
        return self._parse_result(result)


class BlogSectionWriterChain(LLMChain):
    """Chain to prioritize tasks."""

    @classmethod
    def from_llm(cls, llm: BaseLLM, verbose: bool = True) -> LLMChain:
        """Get the response parser."""
        blog_section_writer_template = (
            "You are a blog writer who is writing about {topic}."
            "You should write a specific section of the blog based on the heading and sub headings provided."
            "Heading: {heading}"
            "Sub Headings: {sub_headings}"
            "The section is part of a larger blog post which follows the following outline: {outline}."
            "A summary of previous sections is provided here for reference."
            "Previous Sections Summary: {context}"
            "Make sure that the beginning and the end of each section you write flows seamlessly with the earlier and later sections of the blog"
            "Remember to optimze SEO based on these keywords: {keywords}."
            "Use markdown to format the blog section."
            "Example: "
            "# Heading 1"
            "content"
            "## Sub Heading 1"
            "content"
        )
        prompt = PromptTemplate(
            template=blog_section_writer_template,
            input_variables=[
                "topic",
                "keywords",
                "heading",
                "outline",
                "sub_headings",
                "context",
            ],
        )
        return cls(prompt=prompt, llm=llm, verbose=verbose)


class BlogSummarizerChain(LLMChain):
    @classmethod
    def from_llm(cls, llm: BaseLLM, verbose: bool = True) -> LLMChain:
        blog_summarizer_template = (
            "Summarize a series of blog sections on the topic '{topic}', providing a concise and coherent overview that captures the main points and themes."
            "This summary will be used as context for another agent to write a new blog section."
            "Ensure the summary is comprehensive enough to inform the new section while maintaining brevity."
            "Blog Sections: {blog_sections}"
        )
        prompt = PromptTemplate(
            template=blog_summarizer_template,
            input_variables=[
                "topic",
                "blog_sections",
            ],
        )
        return cls(prompt=prompt, llm=llm, verbose=verbose)


class OutlineRevisionistChain(LLMChain):
    """Chain to prioritize tasks."""

    @classmethod
    def from_llm(cls, llm: BaseLLM, verbose: bool = True) -> LLMChain:
        """Get the response parser."""
        outline_revisionist_template = (
            "Incorporate the suggested changes to the outline, and return the revised outline."
            "Original Outline: {original_outline}"
            "Suggested Changes: {suggested_changes}"
        )
        prompt = PromptTemplate(
            template=outline_revisionist_template,
            input_variables=["original_outline", "suggested_changes"],
        )
        return cls(prompt=prompt, llm=llm, verbose=verbose)


class BlogSectionRevisionistChain(LLMChain):
    @classmethod
    def from_llm(cls, llm: BaseLLM, verbose: bool = True) -> LLMChain:
        blog_section_revisionist_template = (
            "Incorporate the suggested changes to the blog section, and return the revised blog section."
            "Original Blog Section: {original_section}"
            "Suggested Changes: {suggested_changes}"
            "Format your writing using markdown"
        )
        prompt = PromptTemplate(
            template=blog_section_revisionist_template,
            input_variables=["original_section", "suggested_changes"],
        )
        return cls(prompt=prompt, llm=llm, verbose=verbose)


class BlogWriter(Chain, BaseModel):
    """Controller model for the BlogWriter agent."""

    blog_sections: List[str] = Field(default_factory=list)
    blog_ideation_chain: BlogIdeationChain = Field(...)
    outline_writer_chain: OutlineWriterChain = Field(...)
    outline_revisionist_chain: OutlineRevisionistChain = Field(...)
    outline_parser_chain: OutlineParserChain = Field(...)
    blog_section_writer_chain: BlogSectionWriterChain = Field(...)
    blog_summarizer_chain: BlogSummarizerChain = Field(...)
    blog_section_revisionist_chain: BlogSectionRevisionistChain = Field(...)
    # vectorstore: VectorStore = Field(init=False)
    max_iterations: Optional[int] = None

    class Config:
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True

    def topic_ideation(self, topic: str, keywords: str) -> Dict[str, str]:
        """Get the tasks."""
        response = self.blog_ideation_chain.run(topic=topic, keywords=keywords)
        selected_topic = make_user_select_value(response)

        if selected_topic is None:
            selected_topic = self.topic_ideation(topic, keywords)

        return selected_topic

    def write_outline(self, selected_topic: str, keywords: str) -> str:
        """Write the outline."""
        outline = self.outline_writer_chain.run(topic=selected_topic, keywords=keywords)
        outline_being_edited = True

        while outline_being_edited:
            suggested_changes = ask_human_for_feedback(
                "Here's the blog outline suggest any feedback if needed", outline
            )
            if suggested_changes is None:
                outline_being_edited = False
            else:
                outline = self.outline_revisionist_chain.run(
                    original_outline=outline, suggested_changes=suggested_changes
                )

        return outline

    def write_blog(self, outline: str, topic: str, keywords: str) -> str:
        """Write the blog."""
        heading_idx = 0
        json_outline = self.outline_parser_chain.run(outline=outline)
        for heading, sub_headings in json_outline.items():
            context = (
                ""
                if heading_idx == 0
                else self.blog_summarizer_chain.run(
                    topic=topic, blog_sections="\n".join(self.blog_sections)
                )
            )
            section = self.blog_section_writer_chain.run(
                topic=topic,
                keywords=keywords,
                heading=heading,
                outline=outline,
                sub_headings=sub_headings,
                context=context,
            )
            section_being_edited = True
            while section_being_edited:
                suggested_changes = ask_human_for_feedback(
                    "Here's the blog section suggest any feedback if needed", section
                )
                if suggested_changes is None:
                    section_being_edited = False
                else:
                    section = self.blog_section_revisionist_chain.run(
                        original_section=section, suggested_changes=suggested_changes
                    )

            self.blog_sections.append(section)
            heading_idx += 1

        return "\n".join(self.blog_sections)

    def write_to_file(self, blog: str, file_path: str) -> None:
        """Write the blog to a file."""
        with open(file_path, "w") as f:
            f.write(blog)

    @property
    def input_keys(self) -> List[str]:
        return ["topic", "keywords"]

    @property
    def output_keys(self) -> List[str]:
        return []

    def _call(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Run the agent."""
        topic = inputs["topic"]
        keywords = inputs["keywords"]
        selected_topic = self.topic_ideation(topic, keywords)
        outline = self.write_outline(selected_topic, keywords)
        raw_blog = self.write_blog(outline, topic, keywords)
        self.write_to_file(raw_blog, "blog.md")
        return {}

    @classmethod
    def from_llm(cls, llm: BaseLLM, verbose: bool = False, **kwargs) -> "BlogWriter":
        """Initialize the BlogWriter Controller."""
        blog_ideation_chain = BlogIdeationChain.from_llm(llm, verbose=verbose)
        outline_writer_chain = OutlineWriterChain.from_llm(llm, verbose=verbose)
        outline_revisionist_chain = OutlineRevisionistChain.from_llm(
            llm, verbose=verbose
        )
        outline_parser_chain = OutlineParserChain.from_llm(llm, verbose=verbose)

        blog_section_writer_chain = BlogSectionWriterChain.from_llm(
            llm, verbose=verbose
        )
        blog_summarizer_chain = BlogSummarizerChain.from_llm(llm, verbose=verbose)
        blog_section_revisionist_chain = BlogSectionRevisionistChain.from_llm(
            llm, verbose=verbose
        )
        return cls(
            blog_ideation_chain=blog_ideation_chain,
            outline_writer_chain=outline_writer_chain,
            outline_revisionist_chain=outline_revisionist_chain,
            outline_parser_chain=outline_parser_chain,
            blog_section_writer_chain=blog_section_writer_chain,
            blog_summarizer_chain=blog_summarizer_chain,
            blog_section_revisionist_chain=blog_section_revisionist_chain,
            **kwargs,
        )


if __name__ == "__main__":
    verbose = False
    llm = ChatOpenAI(model_name="gpt-4")
    blog_writer = BlogWriter.from_llm(llm, verbose=verbose)
    blog_writer(
        {
            "topic": "AI personal assistants in Whatsapp",
            "keywords": "ai, personal assistants, whatsapp, productivity",
        }
    )
