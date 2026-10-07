$(document).keydown(function(event){
	var charCode = event.keyCode;
	var charStr = String.fromCharCode(charCode);
	statusKeys[charCode] = true;
});

$(document).keyup(function(event){
	var charCode = event.keyCode;

	var charStr = String.fromCharCode(charCode);
	statusKeys[charCode] = false;
});
