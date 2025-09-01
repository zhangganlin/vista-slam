function imageZoom(imgID, resultIDs, n_images) {
    var img, lens, result, cx, cy, x, y;
    var img_ind = 0

    img = document.getElementById(imgID);

    /* Create lens: */
    lens = document.createElement("DIV");

    lens.setAttribute("class", "img-zoom-lens");
    // lens.setAttribute("color", "red");

    /* Insert lens: */
    img.parentElement.insertBefore(lens, img);

    let index = document.getElementById(imgID).getAttribute("src").split("/").at(-1).split(".")[0]
    
    /* Get zoom in image id*/
    var res = []
    for (var i = 0; i < resultIDs.length; i++) {
        res.push(document.getElementById(resultIDs[i]));
        
        /* Calculate the ratio between result DIV and lens: */
        cx = res.at(-1).offsetWidth / lens.offsetWidth;
        cy = res.at(-1).offsetHeight / lens.offsetHeight;

        /* Set background properties for the result DIV */
        res.at(-1).style.backgroundImage = "url('" + res.at(-1).dataset.image + (index).slice(-3) + ".png')";
        res.at(-1).style.backgroundSize = img.width * cx  + "px " + img.height * cy + "px";
    }
    
    cx = res[0].offsetWidth / lens.offsetWidth;
    cy = res[0].offsetHeight / lens.offsetHeight;
    
    /* Execute a function when someone moves the cursor over the image, or the lens: */
    lens.addEventListener("mousemove", moveLens);
    img.addEventListener("mousemove", moveLens);

    /* Change images when click: */
    // lens.addEventListener("click", nextImage);
    // img.addEventListener("click", nextImage);
    
    function moveLens(e) {
        /* Prevent any other actions that may occur when moving over the image */
        e.preventDefault();
        
        // Resize zoomed patches to cope with various resolutions
        for (var i = 0; i < res.length; i++) {
            res[i].style.backgroundSize = img.width * cx  + "px " + img.height * cy + "px";
        }
        
        /* Get the cursor's x and y positions: */
        var pos = getCursorPos(e);
        /* Calculate the position of the lens: */
        x = pos.x - (lens.offsetWidth / 2);
        y = pos.y - (lens.offsetHeight / 2);
        /* Prevent the lens from being positioned outside the image: */
        if (x > (img.width - lens.offsetWidth)) {x = img.width - lens.offsetWidth;}
        if (x < 0) {x = 0;}
        if (y > (img.height - lens.offsetHeight)) {y = img.height - lens.offsetHeight;}
        if (y < 0) {y = 0;}
        /* Set the position of the lens: */
        lens.style.left = img.offsetLeft + x + "px";
        lens.style.top = img.offsetTop + y + "px";
        
        /* Display what the lens "sees": */
        for (var i = 0; i < res.length; i++) {
            res[i].style.backgroundPosition = "-" + (x * cx) + "px -" + (y * cy) + "px"; 
        }
        
    }
    
    function nextImage(e) {
        e.preventDefault();
        
        img_ind = (((img_ind + 1) < n_images) ? img_ind + 1 : 0)
        
        /* Change reference image */
        img.src = img.dataset.image + ("000" + img_ind).slice(-3) + ".png"

        /* Change zoomed in patches*/
        for (var i = 0; i < res.length; i++) {
            res[i].style.backgroundImage = "url('" + res[i].dataset.image + ("000" + img_ind).slice(-3) + ".png')";
        }
    }

    function getCursorPos(e) {
        var a, x = 0, y = 0;
        e = e || window.event;
        /* Get the x and y positions of the image: */
        a = img.getBoundingClientRect();
        /* Calculate the cursor's x and y coordinates, relative to the image: */
        x = e.pageX - a.left;
        y = e.pageY - a.top;
        /* Consider any page scrolling: */
        x = x - window.pageXOffset;
        y = y - window.pageYOffset;
        return {x : x, y : y};
    }
} 
