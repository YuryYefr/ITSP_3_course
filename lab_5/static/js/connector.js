$(document).ready(function () {
    function populateURLField() {
        if (values.length > 0) {
            $('#url').val(values[0]);
        } else {
            $('#url').val('End of values');
            showPopup();
        }
    }

    function refreshValuesList() {
        $('#values-list').text(values.join(", "));
    }

    function showPopup() {
        $('#popup-overlay').fadeIn();
        $('#result-list-popup').fadeIn();
    }

    function hidePopup() {
        $('#popup-overlay').fadeOut();
        $('#result-list-popup').fadeOut();
    }

    // Initial setup
    populateURLField();
    refreshValuesList();

    $('#next').click(function () {
        if (values.length > 0) {
            values.shift();
            populateURLField();
            refreshValuesList();
        }
    });

    $('#crawl').click(function () {
        const url = $('#url').val();
        if (url && url !== 'End of values') {
            $.ajax({
                url: '/extract-name',
                type: 'POST',
                data: JSON.stringify({url: url}),
                contentType: 'application/json',
                success: function (data) {
                    $('#result').text('Extracted Name: ' + data.name);
                    $('#result-list').append('<h1>' + data.name + '</h1>');
                },
                error: function (jqXHR) {
                    $('#result').text('Error: ' + jqXHR.responseJSON.error);
                }
            });
            values.shift();
            populateURLField();
            refreshValuesList();
        }
    });
    $('#restart').click(function () {
        location.reload()
    });

    $('#close-popup').click(function () {
        $('#result').attr('hidden', true)
        hidePopup();
        $('#restart').removeAttr('hidden')
    });
});