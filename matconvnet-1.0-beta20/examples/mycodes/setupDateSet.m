function imdb = setupDateSet(dataset)

opts.dataset = dataset;
opts.dataDir = ['../data/' dataset];
opts.seed = 0;
opts.lite = false;

switch opts.dataset
    case 'scene67', imdb = setupScene67(opts.dataDir,'lite',opts.lite);
    otherwise, error('Unknown dataset type');
end
