const input =
    document.getElementById("messageInput");

const sendButton =
    document.getElementById("sendButton");

const messages =
    document.getElementById("messages");


function addMessage(
    text,
    type
) {
    const message =
        document.createElement("div");

    message.className =
        `message ${type}`;

    const label =
        document.createElement("span");

    label.className = "label";

    label.textContent =
        type === "user"
            ? "YOU"
            : "OMNIX";

    const content =
        document.createElement("div");

    content.textContent = text;

    message.appendChild(label);
    message.appendChild(content);

    messages.appendChild(message);

    window.scrollTo(
        0,
        document.body.scrollHeight
    );

    return content;
}


async function sendMessage() {

    const message =
        input.value.trim();

    if (!message) {
        return;
    }

    input.value = "";

    sendButton.disabled = true;

    const welcome =
        document.querySelector(".welcome");

    if (welcome) {
        welcome.remove();
    }

    addMessage(
        message,
        "user"
    );

    const aiMessage =
        addMessage(
            "Thinking...",
            "ai"
        );

    try {

        const response =
            await fetch(
                "http://127.0.0.1:8000/api/chat",
                {
                    method: "POST",

                    headers: {
                        "Content-Type":
                            "application/json"
                    },

                    body: JSON.stringify({
                        message: message
                    })
                }
            );

        if (!response.ok) {
            throw new Error(
                "Server error"
            );
        }

        const data =
            await response.json();

        aiMessage.textContent =
            data.response || "No response.";

    } catch (error) {

        aiMessage.textContent =
            "Could not connect to OMNIX backend.";

    } finally {

        sendButton.disabled = false;
        input.focus();
    }
}


sendButton.addEventListener(
    "click",
    sendMessage
);


input.addEventListener(
    "keydown",
    event => {

        if (
            event.key === "Enter" &&
            !event.shiftKey
        ) {
            event.preventDefault();
            sendMessage();
        }

    }
);
