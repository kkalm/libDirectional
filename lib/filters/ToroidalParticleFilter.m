classdef ToroidalParticleFilter < AbstractToroidalFilter & HypertoroidalParticleFilter
    % SIR Particle filter on the torus    
    
    methods

        function this = ToroidalParticleFilter(nParticles)
            % Constructor
            %
            % Parameters:
            %   nParticles (integer > 0)
            %       number of particles to use  
            if nargin < 1
                nParticles = 10^3;
            end
            this@HypertoroidalParticleFilter(nParticles,2);
            this.wd = ToroidalWDDistribution(this.wd.d,this.wd.w);
        end

        function setState(this, wd_)
            % Sets the current system state
            %
            % Parameters:
            %   distribution (AbstractToroidalDistribution)
            %       new state            
            assert (isa (wd_, 'AbstractToroidalDistribution'));
            if ~isa(wd_, 'ToroidalWDDistribution')
                wd_ = ToroidalWDDistribution(wd_.sample(length(this.wd.d)));
            end
            this.wd = wd_;
        end

        function updateMixture(this, WD, beta)
            % Updates assuming identity measurement model, i.e.,
            % z(k) = x(k) + v(k)    mod 2pi,
            % where v(k) is additive noise given by noiseDistribution.
            %
            % Parameters:
            %   noiseDistribution (AbstractCircularDistribution)
            %       distribution of additive noise
            %   z (scalar)
            %       measurement in [0, 2pi)
            %   beta (scalar)
            %       mixing coefficient
            if nargin < 3
                beta = 1; % resample all, equivalent to SIR
            end
            % assert(isa (WD, 'ToroidalWDDistribution'));
            assert(isscalar(beta));
            
            numSamples = length(this.wd.d);
            numThis = round(numSamples * beta);
            numPrev = numSamples - numThis;
            
            thisSamples = this.wd.sample(numThis);
            prevSamples = WD.sample(numPrev);
            d = [thisSamples prevSamples];
            this.wd = ToroidalWDDistribution(d); % replace samples and add weights
            
            %this.wd.plot;
        end
        
        function [wd, wd_] = Recall(this, noiseDistribution, z)
            likelihood = LikelihoodFactory.additiveNoiseLikelihood(@(x) x, noiseDistribution);
            wd = this.wd;
            wd_2 = wd.marginalizeTo1D(2); % marginalise over probe dimension
            wd_2 = wd_2.reweigh(@(x) likelihood(z, x));
            % figure(3); M.plot;
            wd.w = wd_2.w;
            % resample after reweighting
            wd_d = wd.sample(length(this.wd.d));
            wd_w = 1/size(this.wd.d,2)*ones(1,size(this.wd.d,2));
            wd_ = ToroidalWDDistribution(wd_d, wd_w);
        end

    end
    
end

