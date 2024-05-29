let popup = document.getElementById('popup');
let overlay = document.getElementById('overlay');
let fileinput = document.getElementById('fileinput');
let xmark = document.getElementById('close');
let uploadbtn = document.getElementById('upload-btn')
let loader = document.querySelector(".loader");

//Open Pop Up window
function openPopup() {
    popup.classList.add('open-popup');
    overlay.classList.add('show-overlay');
}

//Close Pop Up window
function closePopup() {
    popup.classList.remove('open-popup');
    overlay.classList.remove('show-overlay');
    fileinput.reset();
}

//Display loading icon
function openLoading() {
    popup.classList.remove('open-popup');
    loader.classList.add('show-loader')
}

//Get number of form submissions from local storage or cookies
function getFormSubmissionCount() {
    if (localStorage.getItem('formSubmissionCount')) {
        return parseInt(localStorage.getItem('formSubmissionCount'));
    } else {
        localStorage.setItem('formSubmissionCount', '0');
        return 0;
    }
}

function hideItems() {
    submitbtn.classList.add('hide-btn');
    feedback.classList.add('hide-feedback');
}

//Close Pop Up window with escape
document.addEventListener("keydown", function(event) {
    if (event.key === "Escape") {
        closePopup();
    }
});

//Close Pop Up window with x mark icon
xmark.addEventListener("click", closePopup);

//Handle form submissions
fileinput.addEventListener("submit", function(event) {
    const submissionCount = getFormSubmissionCount();
    if (submissionCount >= 3) {
        alert("You have reached your maximum number of uses.");
        event.preventDefault();
    } else {
        const updatedCount = submissionCount + 1;
        localStorage.setItem('formSubmissionCount', updatedCount.toString());
        console.log(localStorage.getItem('formSubmissionCount'));
        openLoading();
    }
});

//Turn off confirm form resubmission
if (window.history.replaceState) {
    window.history.replaceState(null, null, window.location.href);
};