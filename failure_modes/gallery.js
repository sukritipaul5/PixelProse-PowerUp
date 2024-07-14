const galleryData = [
    { fileName: "Sample_1.png", failureMode: "Count/Multiple Objects", prompt: "An arrangement of exactly two red apples and precisely three yellow bananas on a circular plate. Blur background, product photography." },
    { fileName: "Sample_2.png", failureMode: "Colour attribute binding", prompt: "A miniature red sheep driving a white car, Pixar-style 3D rendering, highly detailed." },
    { fileName: "Sample_3.png", failureMode: "Shape attribute binding", prompt: "A surreal landscape featuring a large, pyramid-shaped cloud floating in the sky. Below it, a spherical lake reflects the cloud and sky. The scene has soft, pastel colors. Hyper-realistic rendering, 8k resolution." },
    { fileName: "Sample_4.png", failureMode: "Texture attribute binding", prompt: "An award-winning photo of a cute marble boat with visible veining, floating on a rough sea made entirely of sandpaper." },
    { fileName: "Sample_5.png", failureMode: "Spatial Relation", prompt: "A puppy balanced precariously on the head of a patient dog, studio lighting, high detail, 4K resolution." }
];

const galleryElement = document.querySelector('.gallery');

galleryData.forEach(item => {
    const card = document.createElement('div');
    card.className = 'image-card';
    
    const img = document.createElement('img');
    img.src = `/Users/sukriti/Desktop/SDXL-Inference/Outputs/${item.fileName}`;
    img.alt = item.failureMode;
    
    const failureMode = document.createElement('div');
    failureMode.className = 'failure-mode';
    failureMode.textContent = item.failureMode;
    
    const prompt = document.createElement('div');
    prompt.className = 'prompt';
    prompt.textContent = item.prompt;
    
    card.appendChild(img);
    card.appendChild(failureMode);
    card.appendChild(prompt);
    galleryElement.appendChild(card);
});