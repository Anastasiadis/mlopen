var ret;
var table;
var newTable;

function getCookieVal(name) {
    var cookie = decodeURIComponent(document.cookie);
    var vals = cookie.split(';');
    name += '=';
    var ret;
    vals.forEach(val => {
        while (val.charAt(0) == ' ') {
            val = val.substring(1);
        }
        if (val.indexOf(name) == 0) {
            ret = val.substring((name).length, val.length);
        } else {
            ret = '';
        }
    });
    return ret;
}

const csrftoken = getCookieVal('csrftoken');

function generateTable() {
    newTable = false;
    var arr = [];
    //$('#main_content').html(ret.columns.length);
    for (i = 0; i < ret.columns.length; i++) {
        var temp = {title: JSON.stringify(ret.columns[i])};
        arr.push(temp);
    }

    $('#wait').hide();

    $('#table tbody tr').remove();
    table.destroy();
    $('#table').DataTable( {
        pagingType: "full_numbers",
        data: ret.data,
        columns: [
            {title: "Original"},
            {title: "Sentiment"}
        ]
    } );
    //$('#main_content').html(arr[0].title);
}


function paint(tab){
            table = $('#table').DataTable();
            table.clear();
            table.draw();
            generateTable();
    }


$(document).ready(function(){
    $('#table').hide();
    $('#wait').show();

    $('#submit_btn').click(function(event) {
        event.preventDefault(); id="wait"

        var params = {"pipelines": document.getElementById("pipelines").elements[1].value,
            "input": document.getElementById("files").elements[1].value
        };
        //$('#main_content').html(jQuery.param(params));

        $.ajax({
            type: 'POST',
            url: '.',
            beforeSend: function(request){
                /* eslint-disable no-undef */
                request.setRequestHeader('X-CSRFToken', csrftoken);
                /* eslint-enable no-undef */
            },
            data: jQuery.param(params),//$(this).serialize(),
            success: function(data){
                if (data !== undefined && data !== null){
                    if (!Object.prototype.hasOwnProperty.call(data, 'empty')) {
                        $('#table').show();
                        ret = data;
                        paint();
                        /*
                        $('#testreport_table').DataTable({
                            'pageLength': 25,
                            'order': []
                        });
                        */
                    }
                    else {
                        $('#main_content').html('Weeeeeeeeeeeeeeeeeeeeeeell No updates returned for this specific query. Try a different query.');
                    }
                }
                else{
                    $('#main_content').html('Weeeeeeeeeeeeeeeeeeeeeeeeeeeell Invalid Data Returned by Backend');
                }
            },
            error: function(request){
                var response = JSON.parse(request.responseText);
                $('#spinner_container').hide();
                console.log(response.messages);
                for (var key in response.messages) {
                    if(!Object.prototype.hasOwnProperty.call(response.messages, key)){
                        continue;
                    }
                    $('#id_' + key).addClass('is-invalid');
                    $('#page_content').append('<div class="alert alert-danger" role="alert">' + response.messages[key] + '</div>');
                }

            }
        });
    });

    setTimeout(function(){
        $('#submit_btn').click();
    }, 500);

});