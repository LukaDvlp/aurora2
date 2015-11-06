//!
// Coordinate systems
//
dpm = 50;  //  pix/meter
width  = $("#canvas").width();  // pix
height = $("#canvas").height();  // pix

//var past_xy = [0, 0];
//var dist = 0;

function pix2world(pix) { return pix / dpm; }
function world2pix(meter) { return meter * dpm; }

function x2v(x) { return height / 2 - world2pix(x); }
function y2u(y) { return width / 2 - world2pix(y); }
function v2x(v) { return pix2world(height / 2 - v); }
function u2y(u) { return pix2world(width / 2 - u); }

function rad2deg(rad) { return rad / 3.1415 * 180.0; }
function deg2rad(deg) { return deg * 3.1415 / 180.0; }


//!
// Main loop
//

var flags = new Array();
flags['map'] = "VISUAL";
flags['cam'] = "LEFT";
flags['vision'] = false;
flags['logger'] = false;
flags['adc']    = false;
flags['drive']  = false;

function domReady() {
    //! canvas setting
    ctx = $("#canvas")[0].getContext("2d");
    mouseCB();

    //! switch setting
    $("[name=adc-status]").bootstrapSwitch('onSwitchChange', function() {
        manageProgram("adc", $(this).is(":checked"));
        manageProgram("compass", $(this).is(":checked"));
    });
    $("[name=vision-status]").bootstrapSwitch('onSwitchChange', function() {
        manageProgram("vision", $(this).is(":checked"));
    });
    $("[name=logger-status]").bootstrapSwitch('onSwitchChange', function() {
        manageProgram("logger", $(this).is(":checked"));
    });
    $("[name=drive-status]").bootstrapSwitch('onSwitchChange', function() {
        manageProgram("drive", $(this).is(":checked"));
    });

    $("[name=map-mode]").change(function() { flags['map'] = $(this).val(); });
    $("[name=cam-mode]").change(function() { flags['cam'] = $(this).val(); });

    $("#btn-set-goal").click(function(e) { e.preventDefault(); setGoal(); });
    $("#btn-delete-goal").click(function(e) { e.preventDefault(); deleteGoal(); });
    $("#btn-download-log").click(function() { alert('not implemented'); });

    //! screen update setting
    refleshImages(1);
    refleshMeasurements(1);
    refleshMessages(1);

}

function refleshImages(rate) {
    switch (flags['map']) {
        case "VISUAL":
            $("#map-snapshot").attr("src", getBaseUrl() + "img/_images_visual_map.png?" + Math.random());
            break;
        case "HAZARD":
            $("#map-snapshot").attr("src", getBaseUrl() + "img/_images_hazard_map.png?" + Math.random());
            break;
        case "ELEVATION":
            $("#map-snapshot").attr("src", getBaseUrl() + "img/_images_elev_map.png?" + Math.random());
            break;
    }

    switch (flags['cam']) {
        case "LEFT":
            $("#camera-snapshot").attr("src", "http://192.168.201.61/axis-cgi/jpg/image.cgi?resolution=320x240&" + Math.random());
            //$("#camera-snapshot").attr("src", getBaseUrl() + "img/_images_left.png?" + Math.random());
            break;
        case "RIGHT":
            $("#camera-snapshot").attr("src", "http://192.168.201.62/axis-cgi/jpg/image.cgi?resolution=320x240&" + Math.random());
            break;
        case "DISPARITY":
            $("#camera-snapshot").attr("src", getBaseUrl() + "img/_images_disparity.png?" + Math.random());
            break;

    }

    setTimeout(function() {
        refleshImages(rate);
    }, 1000.0 / rate);
}

function refleshMeasurements(rate) {
    if (flags['adc']) {
        getResource('adc/get_all', function(arg) {
            data = arg.split(" ").map(parseFloat);
            //console.log(data);

            $("#global-pose-roll").text(rad2deg(data[5]).toFixed(1));
            $("#global-pose-pitch").text(rad2deg(data[4]).toFixed(1));
            $("#global-pose-roll").css("color", (Math.abs(rad2deg(data[5])) > 15 ? "red": "black"));
            $("#global-pose-pitch").css("color", (Math.abs(rad2deg(data[4])) > 15 ? "red": "black"));
            $("#img-roll").css("transform", "rotate(" + Math.round(rad2deg(data[5])) + "deg)"); 
            $("#img-pitch").css("transform", "rotate(" + Math.round(rad2deg(data[4])) + "deg)"); 
            

            $("#state-mob-busv").text(data[12].toFixed(1));
            $("#state-com-busv").text(data[13].toFixed(1));
            $("#state-mob-busv").css("color", (data[12] < 28 ? "red": "black"));
            $("#state-com-busv").css("color", (data[13] < 14 ? "red": "black"));

            $("#state-mob-power").text((data[12] * data[14]).toFixed(1));
            $("#state-com-power").text((data[13] * data[15]).toFixed(1));
        });

        getResource('compass/get_all', function(arg) {
            data = arg.split(" ").map(parseFloat);
            $("#global-position-heading").text(rad2deg(data[0]).toFixed(1));
        });
    }

    if (flags['vision']) {
        getResource('vision/pose/get', function(arg) {
            data = arg.split(" ").map(parseFloat);

            $("#global-position-east").text(data[0].toFixed(2));
            $("#global-position-north").text(data[1].toFixed(2));
            //$("#global-position-heading").text(rad2deg(data[1]).toFixed(1));
            $("#global-distance").text(data[3].toFixed(2));

        });
    }

    setTimeout(function() {
        refleshMeasurements(rate);
    }, 1000.0 / rate);
}

function refleshMessages(rate) {
    getResource('message/get', function(arg) {
        if (arg.length > 0) msg(arg);
    });

    setTimeout(function() {
        refleshMessages(rate);
    }, 1000.0 / rate);
}

function manageProgram(type, flag) {
    flags[type] = flag;
    if (flags[type] == true) {
        getResource(type + '/start', function(arg) { });
    } else {
        getResource(type + '/stop', function(arg) { });
    }
}



//!
// Network utility
//

function getResource(resource, callback) {
    $.ajax({
        type: "GET",
        url: resource,
        success: function(result, status, xhr) {
            callback(result);
        },
        error: function(result, status, xhr) {
        }
    });
}









/////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////

function updateScreen() {
    var view_terrain = new Image();
    var view_left = new Image();
    view_terrain.onload = function() {
        $("#alert-comm").hide();
        $("#view_terrain").attr("src", this.src);
        setTimeout(updateScreen, 3000);
    }
    view_terrain.onerror = function() {
        $("#alert-comm").show();
        setTimeout(updateScreen, 1000);
    }
    view_terrain.src = getBaseUrl() + "img/_images_cost_map.png?" + (new Date()).getTime();
    view_left.src = getBaseUrl() + "img/_images_left.png?" + (new Date()).getTime();

    var view_class = new Image();
    view_class.src = getBaseUrl() + "img/_images_cost_map.png?" + (new Date()).getTime();
    view_class.onload = function() {
        $("#view_class").attr("src", this.src);
    }

}

function clearCanvas() {
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
}

function mouseCB() {
    mousedown = false;

    function _setGoalToInput(e) {
        var rect = canvas.getBoundingClientRect();
        u = e.clientX - rect.left;
        v = e.clientY - rect.top;

        $("#goalX").val(v2x(v).toFixed(2));
        $("#goalY").val(u2y(u).toFixed(2));
    }

    canvas.onmousedown = function(e) {
        mousedown = true;
    }
    
    canvas.onmousemove = function(e) {
        if (mousedown == false) return;
        _setGoalToInput(e);
    }

    canvas.onmouseup = function(e) {
        mousedown = false;
        _setGoalToInput(e);
    }
}

function setGoal() {
    $("#alert-path").hide();
    var x = $("#goalX").val();
    var y = $("#goalY").val();
    msg("Setting target to " + [x, y]);
    $.ajax({
        type: "POST",
        url: '/vision/goal/set',
        data: { startU: y2u(0), startV: x2v(0),
                goalU:  y2u(y), goalV:  x2v(x)},
        success: function(result, status, xhr) {
        },
        error: function(result, status, xhr) {
            $("#alert-path").show();
        }
    });
}

function deleteGoal() {
    $("#alert-path").hide();
    getResource("/vision/goal/clear", function() { });
}

function makePath() {
    $("#alert-path").hide();
    $.post('/planning', {
        startU: y2u(0),
        startV: x2v(0),
        goalU: y2u($("#goalY").val()),
        goalV: x2v($("#goalX").val()),
    }).done(function(arg) {
        msg(arg);
        $.ajax({
            url: "{{ url_for('static', filename='data/path.csv') }}?" + (new Date()).getTime(),
            success: function (csvd) {
                waypoints = $.csv2Array(csvd);
                drawPath(waypoints);
            },
            dataType: "text",
            complete: function () {
            }
        });
    }).fail(function() {                        
        $("#alert-path").show();
    });
}

function drawPath(waypoints) {
    clearCanvas();
    ctx.lineCap = "round";
    ctx.globalCompositeOperation = "source-over";
    ctx.strokeStyle = "#FF00FF";
    ctx.fillStyle = ctx.strokeStyle;
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(width / 2, height / 2);
    for (var i = 0; i < waypoints.length; ++i) {
        ctx.lineTo(y2u(waypoints[i][1]), x2v(waypoints[i][0]));
        ctx.stroke();
    }
}


function msg(text) {
    var date = new Date();
    //var date_fmt = date.getHours() + ":" + date.getMinutes() + ":" + date.getSeconds();
    var date_fmt = ('0' + date.getHours()).slice(-2) + ':'
                   + ('0' + (date.getMinutes())).slice(-2) + ':'
                   + ('0' + (date.getSeconds())).slice(-2)
    $("#notification").html(date_fmt + "--" + text + "<br/>" + $("#notification").html());
}

$( domReady ); 
