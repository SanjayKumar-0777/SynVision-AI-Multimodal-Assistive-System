document.addEventListener('DOMContentLoaded', () => {
    const translateBtn = document.getElementById('translateBtn');
    const speakBtn = document.getElementById('speakBtn');
    const resetBtn = document.getElementById('resetBtn');
    const outputArea = document.getElementById('translationOutput');

    translateBtn.addEventListener('click', async () => {
        try {
            const response = await fetch('/translate', { method: 'POST' });
            const data = await response.json();
            if (data.sentence) {
                outputArea.value = data.sentence;
            } else {
                // If buffer was empty but they clicked translate, maybe show nothing or keep old?
                // Requirements say: Display final translated sentence.
                // If empty buffer, it implies empty sentence.
                if (data.sentence === '') {
                   outputArea.value = "No signs detected yet.";
                }
            }
        } catch (error) {
            console.error('Error translating:', error);
            outputArea.value = "Error during translation.";
        }
    });

    speakBtn.addEventListener('click', async () => {
        const textToSpeak = outputArea.value;
        if (!textToSpeak || textToSpeak === "No signs detected yet." || textToSpeak === "Error during translation.") {
            alert("Nothing to speak!");
            return;
        }

        try {
            const response = await fetch('/speak', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text: textToSpeak })
            });
            const data = await response.json();
            console.log('Speak status:', data);
        } catch (error) {
            console.error('Error speaking:', error);
        }
    });

    resetBtn.addEventListener('click', async () => {
        try {
            await fetch('/reset', { method: 'POST' });
            outputArea.value = "";
            console.log('Buffer cleared');
        } catch (error) {
            console.error('Error resetting:', error);
        }
    });
});
