import gradio as gr
import hydra
from hydra.core.config_store import ConfigStore

from autorag import AutoRAGAssistant, AutoRAGConfig


def update_details_button(show_details):
    new_label = "Hide Details" if show_details else "Show Details"
    return gr.update(value=new_label)


def update_show_details(show_details):

    show_details = not show_details
    return show_details


def update_history(history, backup_history):
    print(history)
    print(backup_history)
    tmp = history
    history = backup_history
    backup_history = tmp
    return history, backup_history


def user(user_message, history: list):
    history.append({"role": "user", "content": user_message})
    yield "", history


html_output = """
    <div style="font-weight: bold; font-size: 25px">
        Auto-RAG: Autonomous Retrieval-Augmented Generation for Large Language Models
    </div>
    <div style="font-weight: bold; font-size: 20px">
        Authors: Tian Yu, Shaolei Zhang, and Yang Feng
    </div>
"""


cs = ConfigStore.instance()
cs.store(name="default", node=AutoRAGConfig)


@hydra.main(version_base="1.3", config_path=None, config_name="default")
def main(config: AutoRAGConfig):
    # load assistant
    assistant = AutoRAGAssistant(config)

    # run assistant
    with gr.Blocks() as demo:
        gr.HTML(html_output)
        show_details = gr.State(True)
        backup_history = gr.State([])

        chatbot = gr.Chatbot(
            type="messages",
            label="Auto-RAG",
            height=500,
            placeholder="Ask me anything!",
            show_copy_button=True,
            bubble_full_width=False,
            layout="bubble",
        )
        msg = gr.Textbox()
        with gr.Row():
            toggle_button = gr.Button(f"Hide Details")
            clear_button = gr.Button("Clear")
        toggle_button.click(update_show_details, show_details, show_details).then(
            update_details_button, show_details, toggle_button
        ).then(update_history, [chatbot, backup_history], [chatbot, backup_history])
        msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
            assistant.interactive_answer,
            [chatbot, show_details],
            [chatbot, backup_history],
        )
        clear_button.click(lambda x: [], chatbot, chatbot).then(
            lambda x: [], backup_history, backup_history
        )
    demo.launch(server_name="0.0.0.0")


if __name__ == "__main__":
    main()
