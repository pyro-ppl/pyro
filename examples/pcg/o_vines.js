// ----------------------------------------------------------------------------
// Globals / constants

// TODO: once things are working, strip out the use of 'future,' since we're using immediate mode?
var futurePolicy = 'immediate';
// var futurePolicy = 'lifo';
// var futurePolicy = 'fifo';
// var futurePolicy = 'uniformFromAll';
// var futurePolicy = 'uniformFromDeepest';
// var futurePolicy = 'depthWeighted';
setFuturePolicy(futurePolicy);


var viewport = {xmin: -12, xmax: 12, ymin: -22, ymax: 2};
var norm2world = function(p) {
	return utils.new(THREE.Vector2,
		viewport.xmin + p.x*(viewport.xmax - viewport.xmin), 
		viewport.ymin + p.y*(viewport.ymax - viewport.ymin)
	);	
}

// ----------------------------------------------------------------------------
// Factor encouraging similarity to target image


// Render update
var renderUpdate = function(geo) {
	utils.rendering.drawImgToRenderContext(globalStore.genImg);
	utils.rendering.renderIncr(geo, viewport);
	globalStore.genImg = utils.rendering.copyImgFromRenderContext();
};


// Basically Gaussian log-likelihood, without the constant factor
var makescore = function(val, target, tightness) {
	var diff = val - target;
	return - (diff * diff) / (tightness * tightness);
}

var simTightness = 0.02;
var boundsTightness = 0.001;
var targetFactor = function() {
	renderUpdate(globalStore.geo);
	// Similarity factor
	var sim = utils.normalizedSimilarity(globalStore.genImg, target);
	globalStore.sim = sim;
	var simf = makescore(sim, 1, simTightness);
	// Bounds factors
	var bbox = globalStore.bbox;
	var extraX = (Math.max(viewport.xmin - bbox.min.x, 0) + Math.max(bbox.max.x - viewport.xmax, 0)) / (viewport.xmax - viewport.xmin);
	var extraY = (Math.max(viewport.ymin - bbox.min.y, 0) + Math.max(bbox.max.y - viewport.ymax, 0)) / (viewport.ymax - viewport.ymin);
	var boundsfx = makescore(extraX, 0, boundsTightness);
	var boundsfy = makescore(extraY, 0, boundsTightness);
	var f = simf + boundsfx + boundsfy;
	if (globalStore.prevFactor) {
		factor(f - globalStore.prevFactor);
	} else {
		factor(f);
	}
	globalStore.prevFactor = f;
};


// ----------------------------------------------------------------------------
// The program itself


var makeProgram = function(neurallyGuided) {

	var flipBounds = [ad.scalar.sigmoid];
	var _flip = !neurallyGuided ?
	function(p, localState, name) {
		return sample(Bernoulli({p: p}));
	}
	:
	function(p, localState, name) {
		var guideParams = nnGuide.predict(globalStore, localState, name, flipBounds);
		return sample(Bernoulli({p: p}), {
			guide: Bernoulli({p: guideParams[0]})
		});
	}

	var discrete3Bounds = repeat(3, function() { return ad.scalar.sigmoid; });
	var _discrete3 = !neurallyGuided ?
	function(p0, p1, p2, localState, name) {
		return sample(Discrete({ps: [p0, p1, p2]}));
	}
	:
	function(p0, p1, p2, localState, name) {
		var guideParams = nnGuide.predict(globalStore, localState, name, discrete3Bounds);
		return sample(Discrete({ps: [p0, p1, p2]}), {
			guide: Discrete({ps: guideParams})
		});
	};

	var gmmN = 4;
	var gmmWeightBounds = repeat(gmmN, function() { return ad.scalar.sigmoid; });
	var gmmParamBounds = _.flatten(repeat(gmmN, function() { return [undefined, ad.scalar.exp]; }));
	var gmmBounds = gmmWeightBounds.concat(gmmParamBounds);
	var group = function(lst, n) {
		return (lst.length === 0) ? [] : cons(lst.slice(0, n), group(lst.slice(n), n));
	};
	var unflattenGmmGuideParams = function(guideParams) {
		var weights = guideParams.slice(0, gmmN);
		var params = map(function(musig) { return {mu: musig[0], sigma: musig[1]}; }, group(guideParams.slice(gmmN), 2));
		return {weights: weights, params: params};
	};
	var _gaussian = !neurallyGuided ?
	function(mu, sigma, localState, name) {
		return sample(Gaussian({mu: mu, sigma: sigma}));
	}
	:
	function(mu, sigma, localState, name) {
		var guideParams = nnGuide.predict(globalStore, localState, name, gmmBounds);
		return sample(Gaussian({mu: mu, sigma: sigma}), {
			guide: GaussianMixture(unflattenGmmGuideParams(guideParams))
		});
	};



	var addBranch = function(newbranch, currState) {
		// Update model state
		globalStore.geo = {
			type: 'branch',
			branch: newbranch,
			next: globalStore.geo,
			parent: currState.prevBranch,
			n: globalStore.geo ? globalStore.geo.n + 1 : 1
		};
		globalStore.bbox = globalStore.bbox.clone().union(utils.bboxes.branch(newbranch));

		// Add new heuristic factor
		targetFactor();
	};

	var addLeaf = function(newleaf, currState) {
		// Update model state
		globalStore.geo = {
			type: 'leaf',
			leaf: newleaf,
			next: globalStore.geo,
			parent: currState.prevBranch,
			n: globalStore.geo ? globalStore.geo.n + 1 : 1
		};
		globalStore.bbox = globalStore.bbox.clone().union(utils.bboxes.leaf(newleaf));

		// Add new heuristic factor
		targetFactor();
	};

	var addFlower = function(newflower, currState) {
		// Update model state
		globalStore.geo = {
			type: 'flower',
			flower: newflower,
			next: globalStore.geo,
			parent: currState.prevBranch,
			n: globalStore.geo ? globalStore.geo.n + 1 : 1
		};
		globalStore.bbox = globalStore.bbox.clone().union(utils.bboxes.flower(newflower));

		// Add new heuristic factor
		targetFactor();
	}


	var initialWidth = 0.75;
	var widthDecay = 0.975;
	var minWidthPercent = 0.15;
	var minWidth = minWidthPercent*initialWidth;
	var leafAspect = 2.09859154929577;
	var leafWidthMul = 1.3;
	var flowerRadMul = 1;

	var state = function(obj) {
		return {
			depth: obj.depth,
			pos: obj.pos,
			angle: obj.angle,
			width: obj.width,
			prevBranch: obj.prevBranch,
			features: neurallyGuided ? nnGuide.localFeatures(obj) : undefined
		};
	};

	var polar2rect = function(r, theta) {
		return utils.new(THREE.Vector2, r*Math.cos(theta), r*Math.sin(theta));
	};

	var lOpts = ['none', 'left', 'right'];
	var lProbs = [1, 1, 1];
	var branch = function(currState) {

		// Generate new branch
		var width = widthDecay * currState.width;
		var length = 2;
		var newang = currState.angle + _gaussian(0, Math.PI/8, currState, 'angle');
		var newbranch = {
			start: currState.pos,
			angle: newang,
			width: width,
			end: polar2rect(length, newang).add(currState.pos)
		};
		addBranch(newbranch, currState);

		var newState = state({
			depth: currState.depth + 1,
			pos: newbranch.end,
			angle: newbranch.angle,
			width: newbranch.width,
			prevBranch: globalStore.geo
		});

		// Generate leaf?
		future(function() {
			var leafOpt = lOpts[_discrete3(lProbs[0], lProbs[1], lProbs[2], newState, 'leaf')];
			if (leafOpt !== 'none') {
				var lwidth = leafWidthMul * initialWidth;
				var llength = lwidth * leafAspect;
				var angmean = (leafOpt === 'left') ? Math.PI/4 : -Math.PI/4;
				var langle = newbranch.angle + _gaussian(angmean, Math.PI/12, newState, 'leafAngle');
				var lstart = newbranch.start.clone().lerp(newbranch.end, 0.5);
				var lend = polar2rect(llength, langle).add(lstart);
				var lcenter = lstart.clone().add(lend).multiplyScalar(0.5);
				addLeaf({
					length: llength,
					width: lwidth,
					angle: langle,
					center: lcenter
				}, newState);
			}
		});

		// Generate flower?
		future(function() {
			if (_flip(0.5, newState, 'flower')) {
				addFlower({
					center: newbranch.end,
					radius: flowerRadMul * initialWidth,
					angle: newbranch.angle
				}, newState);
			}
		});

		if (neurallyGuided) {
			nnGuide.step(globalStore, newState);
		}

		// Terminate?
		future(function() {
			var terminateProb = 0.5;
			if (_flip(terminateProb, newState, 'terminate')) {
				globalStore.terminated = true;
			} else {
				// Generate no further branches w/ prob 1/3
				// Generate one further branch w/ prob 1/3
				// Generate two further branches w/ prob 1/3
				future(function() {
					if (!globalStore.terminated && newState.width > minWidth && _flip(0.66, newState, 'branch1')) {
						branch(newState);
						future(function() {
							if (!globalStore.terminated && newState.width > minWidth && _flip(0.5, newState, 'branch2')) {
								branch(newState);
							}
						});
					}
				});
			}
		});
	};

	var generate = function() {
		// Constants needed by the guide architecture
		if (neurallyGuided) {
			nnGuide.constant('target', target);
			nnGuide.constant('viewport', viewport);
			nnGuide.constant('initialWidth', initialWidth);
			nnGuide.constant('minWidth', minWidth);
		}
	
		var w = target.image.width;
		var h = target.image.height;
		globalStore.genImg = utils.new(utils.ImageData2D).fillWhite(w, h);

		if (neurallyGuided) {
			nnGuide.init(globalStore);
		}
		
		globalStore.geo = undefined;
		globalStore.bbox = utils.new(THREE.Box2);

		// Determine starting state by inverting viewport transform
		var starting_world_pos = norm2world(target.startPos);
		var starting_dir = target.startDir;
		var starting_ang = Math.atan2(starting_dir.y, starting_dir.x);

		// These are separated like this so that we can have an initial local
		//    state to feed to the _gaussian for the initial angle.
		var initState = state({
			depth: 0,
			pos: starting_world_pos,
			angle: 0,
			width: initialWidth,
			prevBranch: undefined
		});
		var startState = state({
			depth: initState.depth,
			pos: initState.pos,
			angle: _gaussian(starting_ang, Math.PI/6, initState, 'startAngle'),
			width: initState.width,
			prevBranch: initState.prevBranch
		});

		future(function() { branch(startState); });
		finishAllFutures();

		return globalStore.geo;
	};

	return generate;
}


// ----------------------------------------------------------------------------

// var generate = makeProgram(false);
var generate = makeProgram(true);

// var nParticles = 100;
var nParticles = 30;
// var nParticles = 300;
// var nParticles = 600;
var dist = Infer({method: 'SMC', particles: nParticles, justSample: true, onlyMAP: true}, generate);
var samp = sample(dist);
var ret = {
	viewport: viewport,
	samp: samp
};
ret;




