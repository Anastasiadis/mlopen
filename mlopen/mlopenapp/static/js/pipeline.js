var ret;
var selectedTab = "Graphs";

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
    var cols = [];
    for (const col in ret.columns) {
        cols.push({title: ret.columns[col]});
    }

    $('#table').DataTable( {
        pagingType: "full_numbers",
        data: ret.data,
        columns: cols
    } );
}


function repaint(){
            if (selectedTab == "Graphs"){
                $('#table_wrapper').hide();
                $('#graphs').show();
            }
            else {
                $('#graphs').hide();
                $('#table_wrapper').show();
            }
}


function paint(){
            repaint();
            $('#wait').hide();
            $('#warpper').show();
            if (ret.graphs !== undefined && ret.graphs !== null) {
                var container = document.getElementById('graphs');
                Plotly.newPlot(container, ret.graphs.data, ret.graphs.layout);
                $('#graphs').hide().show(0);
            }
            generateTable();

}


$(document).ready(function(){
    $('#table_wrapper').hide();
    $('#graphs').hide();
    $('#warpper').hide();
    $('#pipeline_results').hide();
    $('#loader').hide();
    $('#pipeline_select').show();
    $('#wait').show();

    $('#submit_btn').click(function(event) {
        event.preventDefault();
        $('#loader').show();

        var params = {"pipelines": document.getElementById("pipelines").elements[1].value,
            "input": document.getElementById("files").elements[1].value
        };

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
                $('#loader').hide();
                if (data !== undefined && data !== null){
                    if (!Object.prototype.hasOwnProperty.call(data, 'empty')) {
                        $('#pipeline_results').show();
                        $('#pipeline_select').hide();
                        ret = data;
                        paint();
                    }
                    else {
                        $('#main_content').html('No updates returned for this specific query. Try a different query.');
                    }
                }
                else{
                    $('#main_content').html('Invalid Data Returned by Backend');
                }
            },
            error: function(request){
                var response = JSON.parse(request.responseText);
                $('#loader').hide();
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
    
    $('input[type=radio][name="group"]').change(function() {
        selectedTab = $(this).val();
        //$('#test').html($(this).val());
        repaint();
    });

    setTimeout(function(){
        $('#submit_btn').click();
    }, 500);

});