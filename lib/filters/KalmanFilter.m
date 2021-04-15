classdef KalmanFilter < AbstractFilter
    % Kalman Filter based on code by
    % Gerhard Kurz, Igor Gilitschenski, Simon Julier, Uwe D. Hanebeck,
    % Recursive Bingham Filter for Directional Estimation Involving 180 Degree Symmetry
    % Journal of Advances in Information Fusion, 9(2):90 - 105, December 2014.    
        
    properties
        gauss
        %dim
    end
    
    methods
        function this = KalmanFilter()
            % Constructor
            this.setState(GaussianDistribution(0,1));
        end
        
        function setState(this, g)
            % Sets the current system state
            %
            % Parameters:
            %   g (GaussianDistribution)
            %       new state
            assert(isa(g, 'GaussianDistribution'));
            this.gauss = g;
            this.dim = size(g.mu,1);
            %this.setCompositionOperator();
        end
        
        function predictIdentity(this, gaussW)           
            % Predicts assuming identity system model, i.e.,
            % x(k+1) = x(k) (+) w(k)    
            % where w(k) is noise given by gaussW.
            % The composition operator (+) refers to a complex or quaternion
            % multiplication.
            %
            % Parameters:
            %   gaussW (GaussianDistribution)
            %       distribution of noise
            assert(isa(gaussW, 'GaussianDistribution'));            
            %assert(abs(norm(gaussW.mu)-1)<1E-5, 'mean must be a unit vector');
            
            % calculates the prective prior distribution
            % p(x_k|x_{k-1})
            % m, P -- mean vector, cov matrix
            % Gaussian
            % p(x_k|x_{k-1}) = N(x_k | A_{k-1}*x_{k-1}, Q)
            
            % mean mu = A * mu;
            mu_ = this.gauss.mu;
            % cov P = A * P * A.' + Q;
            C_ = this.gauss.C + gaussW.C;
            
            this.gauss = GaussianDistribution(mu_, C_);
        end
        
        function updateIdentity(this, gaussV, z)           
            % Updates assuming identity measurement model, i.e.,
            % z(k) = x(k) (+) v(k)    mod 2pi,
            % where v(k) is additive noise given by vmMeas.
            % The composition operator (+) refers to a complex or quaternion
            % multiplication.
            %
            % Parameters:
            %   gaussV (GaussianDistribution)
            %       distribution of additive noise
            %   z (dim x 1 vector)
            %       measurement on the unit hypersphere            
            assert(isa(gaussV, 'GaussianDistribution'));            
            %assert(abs(norm(gaussV.mu)-1)<1E-5, 'mean must be a unit vector');
            %assert(size(gaussV.mu,1) == size(this.gauss.mu,1));
            %assert(all(size(z) == size(this.gauss.mu)));
            %assert(abs(norm(z) - 1) < 1E-5, 'measurement must be a unit vector');
            
            %muVconj = [gaussV.mu(1); -gaussV.mu(2:end)];
            %z = this.compositionOperator(muVconj, z);
            
            %if dot(z,this.gauss.mu)<0 % mirror z if necessary
                %z= -z;
            %end
            
            % Gaussian
            %P = (P^-1 + (1/sigma^2) * H' * H)^-1;
            %m = P * ((1/sigma^2) * H' * y + (P^-1 * m));

            d = this.dim;
            H = eye(d,d); % measurement matrix
            % intermediate variables
            % Innovation covariance of a Kalman/Gaussian filter at step k
            %S = H * P * H.' + R;            
            IS = H*this.gauss.C*H' + gaussV.C; %innovation covariance
            % gain
            %K = P * H.' * S^-1;            
            K = this.gauss.C*H'/IS; % Kalman gain
            % mean
            %m = m + K * (y - H*m);
            IM = z - H*this.gauss.mu; % measurement residual
            mu_ = this.gauss.mu + K*IM; % updated mean
            % cov
            %P = P - K * S * K.';
            C_ = (eye(d,d) - K*H) * this.gauss.C; %updated covariance
            
            %mu_ =mu_/norm(mu_); % enforce unit vector
            
            this.gauss = GaussianDistribution(mu_, C_);
        end
        
        function gauss = getEstimate(this)
            % Return current estimate 
            %
            % Returns:
            %   g(GaussianDistribution)
            %       current estimate
            gauss = this.gauss;
        end
        
        function [mu, C] = getValues(this)
            mu = this.gauss.mu;
            C  = this.gauss.C;
        end
    end
    
end
