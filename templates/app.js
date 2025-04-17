document.getElementById("questionForm").addEventListener("submit", function(event) {
    event.preventDefault();

    const question = document.getElementById("questionInput").value;
    const responseSection = document.getElementById("responseText");

    if (question.trim() === "") {
        alert("Please enter a valid question.");
        return;
    }

    // Disable the form to avoid multiple submissions
    document.querySelector("button").disabled = true;

    // Show loading message
    responseSection.innerHTML = "<p>Loading... Please wait.</p>";

    // Simulate API call to get the response (you would replace this with actual backend call)
    setTimeout(() => {
        // For now, simulate an answer being returned
        const answer = `Here is a detailed answer for your question: "${question}". This is a placeholder answer. The actual response should be fetched from the backend based on the context.`;
        
        // Display the answer in the response section
        responseSection.innerHTML = `<p>${answer}</p>`;

        // Re-enable the form
        document.querySelector("button").disabled = false;
    }, 1500);
});
